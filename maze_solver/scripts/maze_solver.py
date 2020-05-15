#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import matplotlib.pyplot as plt
import queue, collections
import os
import pickle
from termcolor import colored
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_multiply

vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
VMIN = 0.25
VMAX = 2.0
VTURN = 0.25
RADTURN = 0.05
LINACCEL = 0.2
WMIN = np.pi/1.25
WMAX = np.pi/0.25
WACCEL = np.pi/2
MIN_DIAG_LEN = 20

RATE = 100
KP = -30.0
KD = -20

LEN = 0.18
WALL_THICKNESS = 0.01
MOTOR_OFFSET = 0.01			#offset between mouse center and axel center

pose = []
pose_header = []

MAX_FRONT_WALL_DIST = 1.5*LEN
MAX_SIDE_WALL_DIST = 0.8*LEN		#worst case scenario where you can still detect side wall
FRONT_SENSOR_OFFSET = 0.0525
FRONTLR_SENSOR_OFFSET = 0.0385
FRONTLR_SENSOR_COSANGLE = np.cos(0.1)
DIAG_SENSOR_SINANGLE = np.sin(np.pi/3)
FRONT_SENSOR_WALL_DIST = 0.0425		#distance of front wall from laser when mouse is centered in the cell

DIRS = {'N':0,'W':1,'S':2,'E':3, 0:'N', 1:'W', 2:'S', 3:'E'} # bijective mapping to keep it easier to do modular math
COSTS = {'F': 1, 'L': 0.1, 'R': 0.1}

# Globals
maze = []
sensors = []

class Sensors():

	def has_right_wall(self):
		return self.get_sider_distance() < MAX_SIDE_WALL_DIST

	def has_left_wall(self):
		return self.get_sidel_distance() < MAX_SIDE_WALL_DIST

	def has_front_wall(self):
		return self.get_front_distance() < MAX_FRONT_WALL_DIST

	def get_front_error(self):
		fr = self.get_frontr_distance()
		fl = self.get_frontl_distance()
		if fl < MAX_FRONT_WALL_DIST and fr < MAX_FRONT_WALL_DIST:
			return fl - fr
		else:
			return 0.0

	def get_sider_error(self):
		if self.has_right_wall():
			return self.get_sider_distance() - (LEN/2 - WALL_THICKNESS/2)
		else:
			return 0

	def get_sidel_error(self):
		if self.has_left_wall():
			return (LEN/2 - WALL_THICKNESS/2) - self.get_sidel_distance()
		else:
			return 0

	def get_sider_distance(self):
		return self.get_diagr_distance()*DIAG_SENSOR_SINANGLE

	def get_sidel_distance(self):
		return self.get_diagl_distance()*DIAG_SENSOR_SINANGLE

	def get_diagr_distance(self):
		return np.median(self.ranges['DiagR'])

	def get_diagl_distance(self):
		return np.median(self.ranges['DiagL'])

	def get_frontr_distance(self):
		return np.median(self.ranges['FrontR'])*FRONTLR_SENSOR_COSANGLE+FRONTLR_SENSOR_OFFSET

	def get_frontl_distance(self):
		return np.median(self.ranges['FrontL'])*FRONTLR_SENSOR_COSANGLE+FRONTLR_SENSOR_OFFSET

	def get_front_distance(self):
		return np.median(self.ranges['Front']) + FRONT_SENSOR_OFFSET

	def update_wall_counters(self):
		self.wall_counters['Right'] = (self.wall_counters['Right'][0]+self.has_right_wall(), self.wall_counters['Right'][1]+1)
		self.wall_counters['Front'] = (self.wall_counters['Right'][0]+self.has_front_wall(), self.wall_counters['Front'][1]+1)
		self.wall_counters['Left'] = (self.wall_counters['Right'][0]+self.has_left_wall(), self.wall_counters['Left'][1]+1)

	def update_ranges(self, which, val):
		self.ranges[which].append(val)

	def reset(self,wait_for_data=True):
		# clear wall wall counters
		self.wall_counters = {'Front': (0,0), 'Right': (0,0), 'Left': (0,0)}

		# clear ranges history
		self.ranges = {'Front': collections.deque(maxlen=5),
		'FrontR': collections.deque(maxlen=5),
		'FrontL': collections.deque(maxlen=5),
		'DiagR': collections.deque(maxlen=5),
		'DiagL': collections.deque(maxlen=5)}

		# wait till you get valid measurements
		if wait_for_data:
			valid = False
			r = rospy.Rate(RATE)
			while not valid:
				valid = True
				for key in self.ranges.keys():
					valid = valid and len(self.ranges[key]) > 0
				r.sleep()

	def __init__(self):
		self.reset(False)



class GridGraph():

	def print_graph(self):
		print(self.graph)

	def get_path(self,start,goals):
		frontier = queue.PriorityQueue()
		visited = {start:True}
		infrontier = {}

		if start in goals:
			return [], []

		successors = list(self.graph[start])
		for s in successors:
		    if not visited.get(s):
		        frontier.put( (COSTS[self.graph[start][s]], (s,[ self.graph[start][s] ],[s],COSTS[self.graph[start][s]]) ) )
		        infrontier[s] = True

		while not frontier.empty():
			curr_node = frontier.get()
			curr_state = curr_node[1][0]
			curr_actionlist = curr_node[1][1]
			curr_vertexlist = curr_node[1][2]
			curr_cost = curr_node[1][3]

			visited[curr_state] = True
			infrontier[curr_state] = False

			if curr_state in goals:
				return curr_actionlist, curr_vertexlist

			successors = list(self.graph[curr_state])
			for s in successors:
				if not visited.get(s):# and not infrontier.get(s):
					frontier.put( (COSTS[self.graph[curr_state][s]]+curr_cost, (s,curr_actionlist + [ self.graph[curr_state][s] ],curr_vertexlist+[s],COSTS[self.graph[curr_state][s]]+curr_cost)) )
					infrontier[s] = True

		return [],[]

	def remove_edge(self,u,v):
		self.graph[u].pop(v,None)
		self.graph[v].pop(u,None)

	def add_edge(self,u,v,action):
		self.graph[u][v] = action

	def update_edges(self,row,col,dir,rwall,fwall,lwall):
		if dir == 'N':
			if fwall == False and row+1 < self.dim:
				self.add_edge( (row,col,'N'), (row+1,col,'N'), 'F')
				self.add_edge( (row+1,col,'S'), (row,col,'S'), 'F')
			elif row+1 < self.dim:
				self.remove_edge( (row,col,'N'), (row+1,col,'N') )
				self.remove_edge( (row+1,col,'S'), (row,col,'S') )
			if rwall == False and col+1 < self.dim:
				self.add_edge( (row,col,'E'), (row,col+1,'E'), 'F')
				self.add_edge( (row,col+1,'W'), (row,col,'W'), 'F')
			elif col+1 < self.dim:
				self.remove_edge( (row,col,'E'), (row,col+1,'E') )
				self.remove_edge( (row,col+1,'W'), (row,col,'W') )
			if lwall == False and col > 0:
				self.add_edge( (row,col,'W'), (row,col-1,'W'), 'F')
				self.add_edge( (row,col-1,'E'), (row,col,'E'), 'F')
			elif col > 0:
				self.remove_edge( (row,col,'W'), (row,col-1,'W') )
				self.remove_edge( (row,col-1,'E'), (row,col,'E') )
		if dir == 'S':
			if fwall == False and row > 0:
				self.add_edge( (row,col,'S'), (row-1,col,'S'), 'F')
				self.add_edge( (row-1,col,'N'), (row,col,'N'), 'F')
			elif row > 0:
				self.remove_edge( (row,col,'S'), (row-1,col,'S') )
				self.remove_edge( (row-1,col,'N'), (row,col,'N') )
			if rwall == False and col > 0:
				self.add_edge( (row,col,'W'), (row,col-1,'W'), 'F')
				self.add_edge( (row,col-1,'E'), (row,col,'E'), 'F')
			elif col > 0:
				self.remove_edge( (row,col,'W'), (row,col-1,'W') )
				self.remove_edge( (row,col-1,'E'), (row,col,'E') )
			if lwall == False and col+1 < self.dim:
				self.add_edge( (row,col,'E'), (row,col+1,'E'), 'F')
				self.add_edge( (row,col+1,'W'), (row,col,'W'), 'F')
			elif col+1 < self.dim:
				self.remove_edge( (row,col,'E'), (row,col+1,'E') )
				self.remove_edge( (row,col+1,'W'), (row,col,'W') )
		if dir == 'E':
			if fwall == False and col+1 < self.dim:
				self.add_edge( (row,col,'E'), (row,col+1,'E'), 'F')
				self.add_edge( (row,col+1,'W'), (row,col,'W'), 'F')
			elif col+1 < self.dim:
				self.remove_edge( (row,col,'E'), (row,col+1,'E') )
				self.remove_edge( (row,col+1,'W'), (row,col,'W') )
			if rwall == False and row > 0:
				self.add_edge( (row,col,'S'), (row-1,col,'S'), 'F')
				self.add_edge( (row-1,col,'N'), (row,col,'N'), 'F')
			elif row > 0:
				self.remove_edge( (row,col,'S'), (row-1,col,'S') )
				self.remove_edge( (row-1,col,'N'), (row,col,'N') )
			if lwall == False and row+1 < self.dim:
				self.add_edge( (row,col,'N'), (row+1,col,'N'), 'F')
				self.add_edge( (row+1,col,'S'), (row,col,'S'), 'F')
			elif row+1 < self.dim:
				self.remove_edge( (row,col,'N'), (row+1,col,'N') )
				self.remove_edge( (row+1,col,'S'), (row,col,'S') )
		if dir == 'W':
			if fwall == False and col > 0:
				self.add_edge( (row,col,'W'), (row,col-1,'W'), 'F')
				self.add_edge( (row,col-1,'E'), (row,col,'E'), 'F')
			elif col > 0:
				self.remove_edge( (row,col,'W'), (row,col-1,'W') )
				self.remove_edge( (row,col-1,'E'), (row,col,'E') )
			if rwall == False and row+1 < self.dim:
				self.add_edge( (row,col,'N'), (row+1,col,'N'), 'F')
				self.add_edge( (row+1,col,'S'), (row,col,'S'), 'F')
			elif row+1 < self.dim:
				self.remove_edge( (row,col,'N'), (row+1,col,'N') )
				self.remove_edge( (row+1,col,'S'), (row,col,'S') )
			if lwall == False and row > 0:
				self.add_edge( (row,col,'S'), (row-1,col,'S'), 'F')
				self.add_edge( (row-1,col,'N'), (row,col,'N'), 'F')
			elif row > 0:
				self.remove_edge( (row,col,'S'), (row-1,col,'S') )
				self.remove_edge( (row-1,col,'N'), (row,col,'N') )


	def __init__(self,dim):
		self.dim = dim
		self.graph = {}

		for row in range(dim):
			for col in range(dim):

				dir = 'N'
				self.graph[(row,col,dir)] = {}
				self.graph[(row,col,dir)][(row,col,'W')] = 'L'
				self.graph[(row,col,dir)][(row,col,'E')] = 'R'

				dir = 'S'
				self.graph[(row,col,dir)] = {}
				self.graph[(row,col,dir)][(row,col,'E')] = 'L'
				self.graph[(row,col,dir)][(row,col,'W')] = 'R'

				dir = 'E'
				self.graph[(row,col,dir)] = {}
				self.graph[(row,col,dir)][(row,col,'N')] = 'L'
				self.graph[(row,col,dir)][(row,col,'S')] = 'R'

				dir = 'W'
				self.graph[(row,col,dir)] = {}
				self.graph[(row,col,dir)][(row,col,'S')] = 'L'
				self.graph[(row,col,dir)][(row,col,'N')] = 'R'

		# assume maze is empty
		for n in list(self.graph):
			self.update_edges(n[0],n[1],n[2],False,False,False)



class Cell():

	def set_walls(self,n,s,e,w):
		self.north = n
		self.south = s
		self.east = e
		self.west = w
		self.set_visited()

	def set_visited(self,flag=True):
		self.visited = flag

	def is_visited(self):
		return self.visited

	def draw_cell_marker(self,markercolor='r'):
		# rows are y-coords, columns are x-coords
		y = self.row
		x = self.col
		plt.plot([x+0.5],[y+0.5],marker='o',color=markercolor)

	def draw_cell(self):
		# rows are y-coords, columns are x-coords
		y = self.row
		x = self.col

		plt.plot(x,y,marker='o',color='r')
		plt.plot(x+1,y,marker='o',color='r')
		plt.plot(x,y+1,marker='o',color='r')
		plt.plot(x+1,x+1,marker='o',color='r')
		if self.north:
			plt.plot([x, x+1],[y+1, y+1],linewidth=2,color='r')
		if self.south:
			plt.plot([x, x+1],[y, y],linewidth=2,color='r')
		if self.west:
			plt.plot([x, x],[y, y+1],linewidth=2,color='r')
		if self.east:
			plt.plot([x+1, x+1],[y, y+1],linewidth=2,color='r')


	def print_cell(self):
		print(self.row,self.col,self.north,self.west,self.south,self.east)

	def __init__(self,r,c):
		self.north = False
		self.south = False
		self.east = False
		self.west = False
		self.row = r
		self.col = c
		self.visited = False

class Maze():

	def print_maze(self):
		for r in range(self.dim):
			for c in range(self.dim):
				self.cells[r][c].print_cell()

	def prettyprint_maze(self,vertexlist=[]):

		vertexlist = [(v[0],v[1]) for v in vertexlist]

		#os.system('clear')

		# print line by line
		# starting with the northmost wall
		fgcolor = 'red'
		currcolor = 'red'
		pathcolor = 'blue'
		bgcolor = 'on_white'
		attrs = ['bold']
		r = self.dim-1
		lines = colored('o',fgcolor,bgcolor,attrs)
		for c in range(self.dim):
			if self.cells[r][c].north:
				lines += colored('---o',fgcolor,bgcolor,attrs)
			else:
				lines += colored('   o',fgcolor,bgcolor,attrs)
		print(lines)

		for r in reversed(range(self.dim)):
			if self.cells[r][0].west:
				lines = colored('|',fgcolor,bgcolor,attrs)
			else:
				lines = colored(' ',fgcolor,bgcolor,attrs)

			# print current cell with special characters
			# print path with special characters
			for c in range(self.dim):
				color = fgcolor
				if r == self.curr_row and c == self.curr_col:
					cellstr = ' * '
					color = currcolor
				elif (r,c) in vertexlist:
					cellstr = ' . '
					color = pathcolor
				else:
					cellstr = '   '
				if self.cells[r][c].east:
					lines += colored(cellstr,color,bgcolor,attrs)
					lines += colored('|',fgcolor,bgcolor,attrs)
				else:
					lines += colored(cellstr,color,bgcolor,attrs)
					lines += colored(' ',fgcolor,bgcolor,attrs)
			print(lines)
			lines = colored('o',fgcolor,bgcolor,attrs)
			for c in range(self.dim):
				if self.cells[r][c].south:
					lines += colored('---o',fgcolor,bgcolor,attrs)
				else:
					lines += colored('   o',fgcolor,bgcolor,attrs)
			print(lines)

	def draw_maze(self):
		plt.cla()
		for r in range(self.dim):
			for c in range(self.dim):
				self.cells[r][c].draw_cell()
				if self.cells[r][c].is_visited():
					self.cells[r][c].draw_cell_marker('b')

		self.cells[self.curr_row][self.curr_col].draw_cell_marker('g')
		plt.pause(0.01)

	def get_path(self,goal_rows,goal_cols,goal_dirs=None,start_row=None,start_col=None,start_dir=None):
		if start_row == None:
			start_row = self.curr_row
		if start_col == None:
			start_col = self.curr_col
		if start_dir == None:
			start_dir = self.curr_dir
		goals = []
		for i in range(len(goal_rows)):
			if goal_dirs == None:
				goals.append( (goal_rows[i],goal_cols[i],'N') )
				goals.append( (goal_rows[i],goal_cols[i],'S') )
				goals.append( (goal_rows[i],goal_cols[i],'E') )
				goals.append( (goal_rows[i],goal_cols[i],'W') )
			else:
				goals.append( (goal_rows[i],goal_cols[i],goal_dirs[i]) )
		path = self.gridgraph.get_path( (start_row,start_col,start_dir), goals )
		return path

	def update_pose(self,action):
		for a in action:
			if a ==  'F':
				if self.curr_dir == 'N':
					self.set_curr_pose(self.curr_row+1,self.curr_col,self.curr_dir)
				elif self.curr_dir == 'S':
					self.set_curr_pose(self.curr_row-1,self.curr_col,self.curr_dir)
				elif self.curr_dir == 'E':
					self.set_curr_pose(self.curr_row,self.curr_col+1,self.curr_dir)
				elif self.curr_dir == 'W':
					self.set_curr_pose(self.curr_row,self.curr_col-1,self.curr_dir)
			elif a == 'R':
				self.set_curr_pose(self.curr_row,self.curr_col,DIRS[np.mod(DIRS[self.curr_dir]-1,4)])
			elif a == 'L':
				self.set_curr_pose(self.curr_row,self.curr_col,DIRS[np.mod(DIRS[self.curr_dir]+1,4)])
			elif a == 'S':
				self.set_curr_pose(self.curr_row,self.curr_col,self.curr_dir)
			else:
				rospy.logwarn('Wrong action in update_pose')

	def update_cell(self,rwall,fwall,lwall,row=None,col=None,dir=None,visited=True):
		if row == None:
			row = self.curr_row
		if col == None:
			col = self.curr_col
		if dir == None:
			dir = self.curr_dir

		if dir == 'N':
			self.set_wall('E',rwall,row,col)
			self.set_wall('N',fwall,row,col)
			self.set_wall('W',lwall,row,col)
		elif dir == 'S':
			self.set_wall('W',rwall,row,col)
			self.set_wall('S',fwall,row,col)
			self.set_wall('E',lwall,row,col)
		elif dir == 'E':
			self.set_wall('S',rwall,row,col)
			self.set_wall('E',fwall,row,col)
			self.set_wall('N',lwall,row,col)
		elif dir == 'W':
			self.set_wall('N',rwall,row,col)
			self.set_wall('W',fwall,row,col)
			self.set_wall('S',lwall,row,col)

		self.cells[row][col].set_visited(visited)
		self.gridgraph.update_edges(row,col,dir,rwall,fwall,lwall)

	def get_cell(self,row=None,col=None,dir=None):
		if row == None:
			row = self.curr_row
		if col == None:
			col = self.curr_col
		if dir == None:
			dir = self.curr_dir

		rwall = None
		fwall = None
		lwall = None
		if not self.is_visited(row,col):
			rospy.logwarn('Querying walls of unvisited cell')
			return rwall,fwall,lwall

		if dir == 'N':
			rwall = self.get_wall('E',row,col)
			fwall = self.get_wall('N',row,col)
			lwall = self.get_wall('W',row,col)
		elif dir == 'S':
			rwall = self.get_wall('W',row,col)
			fwall = self.get_wall('S',row,col)
			lwall = self.get_wall('E',row,col)
		elif dir == 'E':
			rwall = self.get_wall('S',row,col)
			fwall = self.get_wall('E',row,col)
			lwall = self.get_wall('N',row,col)
		elif dir == 'W':
			rwall = self.get_wall('N',row,col)
			fwall = self.get_wall('W',row,col)
			lwall = self.get_wall('S',row,col)
		return rwall,fwall,lwall


	def is_visited(self,row,col):
		return self.cells[row][col].is_visited()

	def set_wall(self,walldir,exists=True,row=None,col=None):
		if row == None:
			row = self.curr_row
		if col == None:
			col = self.curr_col

		if walldir == 'N':
			self.cells[row][col].north = exists
			if row < self.dim-1:
				self.cells[row+1][col].south = exists
		if walldir == 'S':
			self.cells[row][col].south = exists
			if row > 0:
				self.cells[row-1][col].north = exists
		if walldir == 'E':
			self.cells[row][col].east = exists
			if col < self.dim-1:
				self.cells[row][col+1].west = exists
		if walldir == 'W':
			self.cells[row][col].west = exists
			if col > 0:
				self.cells[row][col-1].east = exists

	def get_wall(self,walldir,row=None,col=None):
		if row == None:
			row = self.curr_row
		if col == None:
			col = self.curr_col

		if walldir == 'N':
			return self.cells[row][col].north
		if walldir == 'S':
			return self.cells[row][col].south
		if walldir == 'E':
			return self.cells[row][col].east
		if walldir == 'W':
			return self.cells[row][col].west

	def set_curr_pose(self,r,c,dir):
		self.curr_row = r
		self.curr_col = c
		self.curr_dir = dir

	def dump(self,filename):
		fileobj = open(filename,'wb+')
		pickle.dump(self,fileobj)

	def __init__(self, dim, init_row=0, init_col=0, init_dir='N'):
		self.dim = dim
		self.cells = [[Cell(r,c) for c in range(dim)] for r in range(dim)]
		self.curr_row = init_row
		self.curr_col = init_col
		self.curr_dir = init_dir
		self.gridgraph = GridGraph(dim)


def get_opt_path(actionlist,vertexlist,maze):
	num_actions = len(actionlist)
	if num_actions < 2:
		return actionlist, vertexlist

	# merge consecutive LL or RR
	merged_actionlist = []
	merged_vertexlist = []
	for curr_action, curr_vertex in zip(actionlist,vertexlist):
		if len(merged_actionlist) == 0:
			merged_actionlist.append(curr_action)
			merged_vertexlist.append(curr_vertex)
		else:
			if curr_action == merged_actionlist[-1] and (curr_action == 'L' or curr_action == 'R'):
				merged_actionlist[-1] += curr_action
			else:
				merged_actionlist.append(curr_action)
				merged_vertexlist.append(curr_vertex)

	# append F after L's and R's
	merged_actionlist2 = []
	merged_vertexlist2 = []
	for curr_action, curr_vertex in zip(merged_actionlist,merged_vertexlist):
		if len(merged_actionlist2) == 0:
			merged_actionlist2.append(curr_action)
			merged_vertexlist2.append(curr_vertex)
		else:
			if curr_action == 'F' and (merged_actionlist2[-1][-1] == 'L' or merged_actionlist2[-1][-1] == 'R'):
				merged_actionlist2[-1] += curr_action
				merged_vertexlist2[-1] = curr_vertex
			else:
				merged_actionlist2.append(curr_action)
				merged_vertexlist2.append(curr_vertex)

	# now merge all the F's together
	# as long as they have been all visited
	merged_actionlist3 = []
	merged_vertexlist3 = []
	for curr_action, curr_vertex in zip(merged_actionlist2,merged_vertexlist2):
		if len(merged_actionlist3) == 0:
			merged_actionlist3.append(curr_action)
			merged_vertexlist3.append(curr_vertex)
		else:
			if curr_action == 'F' and merged_actionlist3[-1] == len(merged_actionlist3[-1])*'F' and maze.is_visited(curr_vertex[0],curr_vertex[1]) and maze.is_visited(merged_vertexlist3[-1][0],merged_vertexlist3[-1][1]):
				merged_actionlist3[-1] += curr_action
				merged_vertexlist3[-1] = curr_vertex
			else:
				merged_actionlist3.append(curr_action)
				merged_vertexlist3.append(curr_vertex)

	# now merge all LF followed by RF and vice versa into a diagonal
	merged_actionlist4 = []
	merged_vertexlist4 = []
	for curr_action, curr_vertex in zip(merged_actionlist3,merged_vertexlist3):
		if len(merged_actionlist4) == 0:
			merged_actionlist4.append(curr_action)
			merged_vertexlist4.append(curr_vertex)
			prev_action = curr_action
		else:
			if curr_action == 'LF' and prev_action == 'RF' and maze.is_visited(curr_vertex[0],curr_vertex[1]) and maze.is_visited(merged_vertexlist4[-1][0],merged_vertexlist4[-1][1]):
				merged_actionlist4[-1] += curr_action
				merged_vertexlist4[-1] = curr_vertex
				prev_action = curr_action
			elif curr_action == 'RF' and prev_action == 'LF' and maze.is_visited(curr_vertex[0],curr_vertex[1]) and maze.is_visited(merged_vertexlist4[-1][0],merged_vertexlist4[-1][1]):
				merged_actionlist4[-1] += curr_action
				merged_vertexlist4[-1] = curr_vertex
				prev_action = curr_action
			else:
				merged_actionlist4.append(curr_action)
				merged_vertexlist4.append(curr_vertex)
				prev_action = curr_action

	return merged_actionlist4, merged_vertexlist4




def odom_callback(msg):
	global pose, pose_header
	pose = msg.pose.pose
	pose_header = msg.header



def scan_callback(msg):
	global sensors
	if msg.header.frame_id == 'sensor_laser':
		sensors.update_ranges('DiagR',msg.ranges[0])
		sensors.update_ranges('Front',msg.ranges[1])
		sensors.update_ranges('DiagL',msg.ranges[2])
	elif msg.header.frame_id == 'sensor_laser_lf':
		sensors.update_ranges('FrontL',msg.ranges[0])
	elif msg.header.frame_id == 'sensor_laser_rf':
		sensors.update_ranges('FrontR',msg.ranges[0])


def turn(action,num_steps=1):
	vel_msg = Twist()

	if action=='L':
		sign = 1
	else:
		sign = -1
	vel_msg.linear.x = 0
	vel_msg.angular.z = sign*WMIN
	r = rospy.Rate(RATE) # hz

	prev_inv_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, -pose.orientation.w]
	angle = 0
	time_delay = 0
	while abs(angle)+(1/RATE+time_delay)*abs(vel_msg.angular.z) < num_steps*np.pi/2 - np.pi/24*num_steps:

		quat = quaternion_multiply([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],prev_inv_orientation)
		angle = (euler_from_quaternion(quat)[2])
		#angle += (euler_from_quaternion(quat)[2])
		#prev_inv_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, -pose.orientation.w]

		time_delay = (rospy.Time.now() - pose_header.stamp).to_sec()

		asign = 0.0
		if abs(angle) > num_steps/2.0*np.pi/2:
			asign = -1.0
		elif abs(angle) < num_steps/2.0*np.pi/2:
			asign = 1.0
		else:
			asign = 0.0
		if abs(vel_msg.angular.z)+asign*WACCEL/RATE >= WMIN and abs(vel_msg.angular.z)+asign*WACCEL/RATE <= WMAX:
			vel_msg.angular.z += sign*(asign*WACCEL)/RATE
		vel_pub.publish(vel_msg)
		r.sleep()


def stop():
	vel_pub.publish(Twist())


def diagonal(startturn,endturn,num_steps=1):

	r = rospy.Rate(RATE)
	vel_msg = Twist()
	vel_msg.linear.x = VTURN
	radius = RADTURN			# radius of the turn
	t = (np.pi*radius/4.0)/VTURN		# how much time to complete the turn

	if startturn == 'L':
		vel_msg.angular.z = (np.pi/4)/t
	else:
		vel_msg.angular.z = -(np.pi/4)/t

	prev_inv_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, -pose.orientation.w]
	angle = 0
	time_delay = 0
	while angle + (1/RATE+time_delay)*abs(vel_msg.angular.z) < np.pi/4-np.pi/30:
		quat = quaternion_multiply([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],prev_inv_orientation)
		angle += abs(euler_from_quaternion(quat)[2])
		prev_inv_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, -pose.orientation.w]
		time_delay = (rospy.Time.now() - pose_header.stamp).to_sec()

		vel_pub.publish(vel_msg)
		r.sleep()
	#stop()

	vel_msg.angular.z = 0
	dist = 0
	prev_pos = pose.position
	prev_err = 0
	targetdist = 1*((LEN/np.sqrt(2))*(num_steps-2) + 0.22)

	ltrigger = -1
	rtrigger = -1
	time_delay = 0
	while dist + (1/RATE+time_delay)*abs(vel_msg.linear.x) <= targetdist:

		time_delay = (rospy.Time.now() - pose_header.stamp).to_sec()

		# keep track of the distance traveled
		dist = predict_dist(dist,prev_pos,pose.position)
		prev_pos = pose.position

		# only when you are not about to reach
		use_ld = True
		use_rd = True
		use_lf = True
		use_rf = True

		if dist > targetdist - LEN/np.sqrt(2) and endturn == 'R':
			use_lf = False
			use_ld = False
		if dist > targetdist - LEN/np.sqrt(2) and endturn == 'L':
			use_rf = False
			use_rd = False
		if dist >= targetdist - 0.11:
			use_lf = False
			use_ld = False
			use_rd = False
			use_rf = False

		err = 0
		if sensors.get_diagr_distance() < LEN/2 and use_rd:
			err += (sensors.get_diagr_distance() - LEN/2)
		if sensors.get_diagl_distance() < LEN/2 and use_ld:
			err += (LEN/2 - sensors.get_diagl_distance())
		if sensors.get_frontr_distance() < LEN and use_rf:
			err += (sensors.get_frontr_distance() - LEN)
		if sensors.get_frontl_distance() < LEN and use_lf:
			err += (LEN - sensors.get_frontl_distance())

		# reset the distance when you notice a high-to-low and then low-to-high transition
		# check if we need to do a high-to-low transition
		# only do it for the inner wall

		print(sensors.get_diagl_distance())
		print(sensors.get_diagr_distance())
		print()
		if ltrigger == -1 and sensors.get_diagl_distance() < 6*LEN/10:
			ltrigger = 0
		# now check if there is a low-to-high transition
		if ltrigger == 0 and sensors.get_diagl_distance() > 7.5*LEN/10:
			ltrigger = -1
			#print("Transition")
			#print "{0:.2f}".format(dist-0.11)
			dist = max(round((dist-0.11)/(LEN*np.sqrt(2))),0)*(LEN*np.sqrt(2))+0.11
			stop()
			rospy.sleep(2)
			#print "{0:.2f}".format(dist-0.11)
			#print "{0:.2f}".format(dist)
			#print "{0:.2f}\n".format(targetdist)
		if rtrigger == -1 and sensors.get_diagr_distance() < 6*LEN/10:
			rtrigger = 0
		# now check if there is a low-to-high transition
		if rtrigger == 0 and sensors.get_diagr_distance() > 7.5*LEN/10:
			rtrigger = -1
			#print("Transition")
			#print "{0:.2f}".format(dist-0.11)
			dist = max(round((dist-0.11)/(LEN*np.sqrt(2))),0)*(LEN*np.sqrt(2))+0.11
			stop()
			rospy.sleep(2)
			#print "{0:.2f}".format(dist-0.11)
			#print "{0:.2f}".format(dist)
			#print "{0:.2f}\n".format(targetdist)


		if dist < targetdist/2:
			vel_msg.linear.x += LINACCEL/RATE
		else:
			vel_msg.linear.x -= LINACCEL/RATE		# TODO hack
		vel_msg.linear.x = min(max(vel_msg.linear.x,VTURN),VMAX)

		vel_msg.angular.z = err*KP
		prev_err = err
		vel_pub.publish(vel_msg)
		r.sleep()
	#stop()

	radius = RADTURN			# radius of the turn
	t = (np.pi*radius/4.0)/VTURN		# how much time to complete the turn
	vel_msg.linear.x = VTURN
	if endturn == 'L':
		vel_msg.angular.z = (np.pi/4)/t
	else:
		vel_msg.angular.z = -(np.pi/4)/t

	prev_inv_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, -pose.orientation.w]
	angle = 0
	time_delay = 0
	while angle + (1/RATE+time_delay)*abs(vel_msg.angular.z) < np.pi/4-np.pi/30:
		time_delay = (rospy.Time.now() - pose_header.stamp).to_sec()

		quat = quaternion_multiply([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],prev_inv_orientation)
		angle += abs(euler_from_quaternion(quat)[2])
		prev_inv_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, -pose.orientation.w]

		vel_pub.publish(vel_msg)
		r.sleep()



def smoothturn(action,rwall=None,fwall=None,lwall=None):
	# this replaces a 1/2F R 1/2F or 1/2F L 1/2F combo

	vel_msg = Twist()
	if action=='L':
		sign = 1
	else:
		sign = -1

	radius = RADTURN			# radius of the turn
	t = (np.pi*radius/2.0)/VTURN		# how much time to complete the turn

	straight_for_smoothturn( (LEN/2-radius+MOTOR_OFFSET), +1, rwall, fwall, lwall)
	r = rospy.Rate(RATE)
	prev_inv_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, -pose.orientation.w]
	angle = 0
	time_delay = 0
	while angle + (1/RATE+time_delay)*abs(vel_msg.angular.z) < np.pi/2-np.pi/12:
		time_delay = (rospy.Time.now() - pose_header.stamp).to_sec()

		quat = quaternion_multiply([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],prev_inv_orientation)
		angle += abs(euler_from_quaternion(quat)[2])
		prev_inv_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, -pose.orientation.w]

		vel_msg.linear.x = VTURN
		vel_msg.angular.z = sign*(np.pi/2)/t
		vel_pub.publish(vel_msg)
		r.sleep()

	rwall, fwall, lwall = straight_for_smoothturn( (LEN/2-radius), +1)
	return rwall, fwall, lwall


def get_corrected_dist(fwall, dist, sensors):
	if fwall and sensors.has_front_wall():
		#print "Forward correction:"
		#print "{0:.3f}".format(dist)
		dist = LEN-sensors.get_front_distance()
		#print "{0:.3f}".format(dist)
		#print "\n"
		return dist
	else:
		return dist


def get_steering_error(rcorr,fcorr,lcorr,sensors):
	err = 0
	if lcorr and rcorr:
		err = (sensors.get_sider_error()+sensors.get_sidel_error())/2
	elif rcorr:	# if right wall exists
		err = sensors.get_sider_error()
	elif lcorr:	# if left wall exists
		err = sensors.get_sidel_error()

	if fcorr:	# if front wall exists
		err += sensors.get_front_error()/2

	return err


def check_for_transition(rwall,lwall,dist,sensors):

	wall_to_nowall = False

	# check for a wall to no wall transition
	# if you do, round off the distance traveled
	if (rwall == True and not sensors.has_right_wall()) or (lwall == True and not sensors.has_left_wall()):
		#print "Wall to no wall transition"
		#print "Before: {0:.2f}".format(dist)
		dist = round(dist/(LEN/2))*LEN/2
		#print "After: {0:.2f}".format(dist)
		wall_to_nowall = True

	# check for no wall to wall transition
	# if you do, we will set the wall flag and wait for wall to no wall transition
	if (rwall == False and sensors.has_right_wall()):
		#print "No wall to wall transition"
		rwall = True
	if (lwall == False and sensors.has_left_wall()):
		#print "No wall to wall transition"
		lwall = True

	return wall_to_nowall,dist,rwall,lwall


def predict_dist(dist,prev_position,curr_position):
	return dist + np.sqrt( (prev_position.x - curr_position.x)**2 + (prev_position.y - curr_position.y)**2 )


def halfforward(rwall=None,fwall=None,lwall=None,vstart=VMIN,accel=0):

	sensors.reset()

	vel_msg = Twist()
	vel_msg.linear.x = vstart

	r = rospy.Rate(RATE) # hz
	dist = 0.0
	prev_err = 0.0
	prev_pos = pose.position

	lwall_hist = []
	fwall_hist = []
	rwall_hist = []

	rwall_new = None
	fwall_new = None
	lwall_new = None
	time_delay = 0
	while dist+(1/RATE+time_delay)*abs(vel_msg.linear.x) < LEN/2 or dist <= LEN/8:
		time_delay = (rospy.Time.now() - pose_header.stamp).to_sec()

		# keep track of the distance traveled
		dist = predict_dist(dist,prev_pos,pose.position)
		prev_pos = pose.position

		wall_to_nowall,dist,rwall,lwall = check_for_transition(rwall,lwall,dist,sensors)
		if wall_to_nowall:
			break

		# record wall history
		rwall_hist.append(sensors.has_right_wall())
		fwall_hist.append(sensors.has_front_wall())
		lwall_hist.append(sensors.has_left_wall())

		# check if we've found a wall
		if rwall == None and rwall_new == None and dist > 0.02 and len(rwall_hist) > 10 and (float(rwall_hist.count(True)+1.0)/float(rwall_hist.count(False)+1.0) > 2 or float(rwall_hist.count(False)+1.0)/float(rwall_hist.count(True)+1.0) > 2):
			rwall_new = rwall_hist.count(True) > rwall_hist.count(False)
		if lwall == None and lwall_new == None and dist > 0.02 and len(lwall_hist) > 10 and (float(lwall_hist.count(True)+1.0)/float(lwall_hist.count(False)+1.0) > 2 or float(lwall_hist.count(False)+1.0)/float(lwall_hist.count(True)+1.0) > 2):
			lwall_new = lwall_hist.count(True) > lwall_hist.count(False)
		if fwall == None and fwall_new == None and dist > 0.02 and len(fwall_hist) > 10 and (float(fwall_hist.count(True)+1.0)/float(fwall_hist.count(False)+1.0) > 2 or float(fwall_hist.count(False)+1.0)/float(fwall_hist.count(True)+1.0) > 2):
			fwall_new = fwall_hist.count(True) > fwall_hist.count(False)

		# Adjust the distance only in the first half of a cell
		# under the assumption that we have already found a wall in the previous one
		dist = get_corrected_dist(fwall,dist,sensors)

		# lateral correction
		rcorr = (rwall == True or rwall_new == True)
		fcorr = (fwall == True or fwall_new == True)
		lcorr = (lwall == True or lwall_new == True)
		err = get_steering_error(rcorr,fcorr,lcorr,sensors)

		# adjust speed
		vel_msg.linear.x += accel/RATE
		vel_msg.linear.x = min(max(vel_msg.linear.x,VMIN),VMAX)

		vel_msg.angular.z = err*KP + (err-prev_err)*KD
		prev_err = err
		vel_pub.publish(vel_msg)

		r.sleep()

	rwall= rwall_hist.count(True)>rwall_hist.count(False)
	fwall = fwall_hist.count(True)>fwall_hist.count(False)
	lwall = lwall_hist.count(True)>lwall_hist.count(False)

	return rwall, fwall, lwall, vel_msg.linear.x


def straight_for_smoothturn(targetdist,dir,rwall=None,fwall=None,lwall=None):

	sensors.reset()

	vel_msg = Twist()
	vel_msg.linear.x = dir*VMIN

	r = rospy.Rate(RATE) # hz
	dist = 0
	prev_pos = pose.position
	prev_err = 0

	lwall_hist = []
	fwall_hist = []
	rwall_hist = []

	rwall_new = None
	fwall_new = None
	lwall_new = None
	time_delay = 0
	while (dist + (1/RATE+time_delay)*abs(vel_msg.linear.x) <= targetdist):
		time_delay = (rospy.Time.now() - pose_header.stamp).to_sec()

		# keep track of the distance traveled
		dist = predict_dist(dist,prev_pos,pose.position)
		prev_pos = pose.position

		# record wall history
		rwall_hist.append(sensors.has_right_wall())
		fwall_hist.append(sensors.has_front_wall())
		lwall_hist.append(sensors.has_left_wall())

		# check if we've found a wall
		if rwall == None and rwall_new == None and dist > 0.02 and len(rwall_hist) > 10 and (float(rwall_hist.count(True)+1.0)/float(rwall_hist.count(False)+1.0) > 2 or float(rwall_hist.count(False)+1.0)/float(rwall_hist.count(True)+1.0) > 2):
			#print("Setting wall")
			rwall_new = rwall_hist.count(True) > rwall_hist.count(False)
		if lwall == None and lwall_new == None and dist > 0.02 and len(lwall_hist) > 10 and (float(lwall_hist.count(True)+1.0)/float(lwall_hist.count(False)+1.0) > 2 or float(lwall_hist.count(False)+1.0)/float(lwall_hist.count(True)+1.0) > 2):
			#print("Setting wall")
			lwall_new = lwall_hist.count(True) > lwall_hist.count(False)
		if fwall == None and fwall_new == None and dist > 0.02 and len(fwall_hist) > 10 and (float(fwall_hist.count(True)+1.0)/float(fwall_hist.count(False)+1.0) > 2 or float(fwall_hist.count(False)+1.0)/float(fwall_hist.count(True)+1.0) > 2):
			#print("Setting wall")
			fwall_new = fwall_hist.count(True) > fwall_hist.count(False)

		# Adjust the distance only in the first half of a cell
		# under the assumption that we have already found a wall in the previous one
		dist = get_corrected_dist(fwall, dist, sensors)

		# lateral correction
		rcorr = (rwall == True or rwall_new == True)
		fcorr = (fwall == True or fwall_new == True)
		lcorr = (lwall == True or lwall_new == True)
		err = get_steering_error(rcorr,fcorr,lcorr,sensors)

		#print "Error: {0:.3f}".format(err)
		#print "Angular velocity: {0:.2f}".format(vel_msg.angular.z)
		#print ranges
		vel_msg.angular.z = err*KP + (err-prev_err)*KD
		prev_err = err
		vel_pub.publish(vel_msg)
		r.sleep()

	rwall= rwall_hist.count(True)>rwall_hist.count(False)
	fwall = fwall_hist.count(True)>fwall_hist.count(False)
	lwall = lwall_hist.count(True)>lwall_hist.count(False)
	return rwall,fwall,lwall


# assumes you are starting in the middle of the cell
# uses front sensor to sense right, front, and left walls
# rotates by pi/4 to sense these walls
# finds an opening in the wall and moves half a cell
# call this function before starting navigation
def make_first_move(maze):

	# find an open wall
	# move for half a cell
	action = ''
	while sensors.get_front_distance() < LEN/2:
		turn('L',1)
		stop()
		sensors.reset(True)
		rospy.sleep(0.5)
		action += 'L'

	action += 'F'
	r,f,l,_ = halfforward()
	stop()
	maze.update_pose(action)
	maze.update_cell(r,f,l)
	return r,f,l


def execute_action(action,r=None,f=None,l=None):

	# forwards can be a long string of F's
	if action == len(action)*'F':
		num_steps = len(action)
		v = VMIN
		# sensor readings are unreliable in the second half of a cell
		ignoresensors = False
		for i in range(num_steps):
			if ignoresensors:
				r = None
				f = None
				l = None
			r,f,l,v = halfforward(r,f,l,v,LINACCEL)
			ignoresensors = not ignoresensors

		# decelerate when the distance to travel is short enough
		# that we may just about reach VMIN
		for i in range(num_steps):
			if ignoresensors:
				r = None
				f = None
				l = None
			if (VMIN*VMIN-v*v)/2.0/(-LINACCEL) > (num_steps-i)*LEN/2:	#TODO deceleration should be faster
				accel = -3*LINACCEL
			else:
				accel = 0.0
			r,f,l,v = halfforward(r,f,l,v,accel)
			ignoresensors = not ignoresensors
	elif action == 'RF':
		r,f,l = smoothturn('R',r,f,l)
	elif action == 'LF':
		r,f,l = smoothturn('L',r,f,l)
	elif action == 'RRF':
		halfforward(r,f,l)
		turn('R',2)
		straight_for_smoothturn(MOTOR_OFFSET,-1)
		halfforward()
		r = None
		f = None
		l = None
	elif action == 'LLF':
		halfforward(r,f,l)
		turn('L',2)
		straight_for_smoothturn(MOTOR_OFFSET,-1)
		halfforward()
		r = None
		f = None
		l = None
	elif action == 'L':
		halfforward(r,f,l)
		turn('L')
		r = None
		f = None
		l = None
	elif action == 'R':
		halfforward(r,f,l)
		turn('R')
		r = None
		f = None
		l = None
	elif action == 'LL':
		halfforward(r,f,l)
		turn('L',2)
		r,f,l,_ = halfforward()
		r = None
		f = None
		l = None
	elif action == 'RR':
		halfforward(r,f,l)
		turn('R',2)
		r,f,l,_ = halfforward()
		r = None
		f = None
		l = None
	elif (action[0] == 'L' or action[0] == 'R') and (action[1] == 'F') and (len(action) % 2 == 0 and len(action) > 2):
		if len(action)/2 > MIN_DIAG_LEN:
			diagonal(action[0],action[-2],len(action)/2)
		else:
			for i in range(len(action)):
				if action[2*i:2*i+2] == 'RF':
					r,f,l = smoothturn('R',r,f,l)
				elif action[2*i:2*i+2] == 'LF':
					r,f,l = smoothturn('L',r,f,l)
		r = None
		f = None
		l = None
	elif action == 'S':
		stop()
		r = None
		f = None
		l = None
	#stop()

	maze.update_pose(action)
	if not (r == None or f == None or l == None):
		maze.update_cell(r,f,l)
	else:
		# instead of reading from sensors, we will just read the actual walls
		# if the previous cell was visited earlier
		r, f, l = maze.get_cell()
	return r,f,l


def wallfollow(maze):
	r,f,l,_ = halfforward()
	stop()
	maze.update_pose('F')
	maze.update_cell(r,f,l)
	maze.prettyprint_maze()

	for i in range(1000):
		if r == False and f == False:
			if np.random.random() > 0.2:
				action = 'F'
			else:
				action = 'RF'
		elif f == False and l == False:
			if np.random.random() > 0.2:
				action = 'F'
			else:
				action = 'LF'
		elif r == False and l == False:
			if np.random.random() > 0.5:
				action = 'RF'
			else:
				action = 'LF'
		elif r == False:
			action = 'RF'
		elif l == False:
			action = 'LF'
		elif f == False:
			action = 'F'
		else:
			action = 'RRF'
		r,f,l = execute_action(action,r,f,l)
		maze.update_pose(action)
		if not (r == None or f == None or l == None):
			maze.update_cell(r,f,l)
		maze.prettyprint_maze()


def goto(goal_rows,goal_cols,goal_dirs=None):

	r,f,l = make_first_move(maze)
	actionlist,vertexlist = maze.get_path(goal_rows,goal_cols,goal_dirs)
	actionlist,vertexlist = get_opt_path(actionlist,vertexlist,maze)

	maze.prettyprint_maze(vertexlist)
	starttime = rospy.Time.now()
	while len(actionlist) > 0:
		action = actionlist.pop(0)
		r,f,l = execute_action(action,r,f,l)
		actionlist,vertexlist = maze.get_path(goal_rows,goal_cols,goal_dirs)
		actionlist,vertexlist = get_opt_path(actionlist,vertexlist,maze)

		maze.prettyprint_maze(vertexlist)
		print(maze.curr_row,maze.curr_col,maze.curr_dir)
		print(r,f,l)
		print (rospy.Time.now()-starttime).to_sec()
		#stop()
	if action[-1] == 'F':
		halfforward(r,f,l)
	stop()


def test(maze):
	print(maze.get_path([0],[0]))

def calibrate(maze):
	global LINACCEL, VMIN, VMAX, RADTURN, VTURN

	r,f,l = make_first_move(maze)
	stop()

	for i in range(6):
		rospy.sleep(0.5)
		turn('R',1)
		stop()

	r,f,l = make_first_move(maze)
	stop()

	for i in range(2):
		rospy.sleep(0.5)
		turn('R',1)
		stop()

	r,f,l = make_first_move(maze)
	stop()

	for i in range(3):
		rospy.sleep(0.5)
		turn('R',2)
		stop()

	r,f,l = make_first_move(maze)
	stop()


def search(maze):
	global LINACCEL, VMIN, VMAX, VTURN, MIN_DIAG_LEN
	for i in range(10):
		goto([7,7,8,8],[7,8,7,8])
		maze.dump('maze.mz')
		rospy.sleep(2)
		goto([0],[0])
		rospy.sleep(2)


def init():
	global sensors

    # Starts a new node
	rospy.init_node('vayu', anonymous=False)
	rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1)
	rospy.Subscriber('/scanlf', LaserScan, scan_callback, queue_size=1)
	rospy.Subscriber('/scanrf', LaserScan, scan_callback, queue_size=1)
	rospy.Subscriber('/odom', Odometry, odom_callback, queue_size=1)

	sensors = Sensors()
	sensors.reset(True)	# wait till subscribers are set up

	maze = Maze(16,0,0)
	#maze = pickle.load(open('maze.mz','rb'))
	maze.set_curr_pose(0,0,'N')

	return maze


if __name__ == '__main__':
	maze = init()
	search(maze)
	#wallfollow(maze)
	#calibrate(maze)
	#test(maze)
