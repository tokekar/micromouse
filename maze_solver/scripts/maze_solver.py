#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import matplotlib.pyplot as plt
import queue, collections
import os
from termcolor import colored
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_multiply

vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
VMIN = 0.1
VMAX = 0.1
LINACCEL = 0.1
WMIN = np.pi/4
WMAX = np.pi/2
WACCEL = np.pi/4

RATE = 100
KP = -20.0
KD = 10
fwd_angle = 60

LEN = 0.18
FWD_REF_DIST = LEN/2-0.065		#distance of front wall from laser when mouse is centered in the cell
MAX_SIDE_WALL_DIST = 0.8*LEN		#worst case scenario where you can still detect side wall, second number is the mouse width
MOTOR_OFFSET = 0.01			#offset between mouse center and axel center
MAX_FRONT_WALL_DIST = 1.2*LEN

rangeshist = [collections.deque(maxlen=5), collections.deque(maxlen=5), collections.deque(maxlen=5)]
robustranges = [0,0,0]
ranges = []
walls = [False, False, False]
pose = []
wallstonowalls = []
nowallstowalls = []

DIRS = {'N':0,'W':1,'S':2,'E':3, 0:'N', 1:'W', 2:'S', 3:'E'} # bijective mapping to keep it easier to do modular math
COSTS = {'F': 1, 'L': 0.1, 'R': 0.1}

# Maze
maze = []
dir = 0


class GridGraph():

	def print_graph(self):
		print(self.graph)

	def get_path(self,start,goal):
		frontier = queue.PriorityQueue()
		visited = {start:True}
		infrontier = {}

		successors = list(self.graph[start])
		for s in successors:
		    if not visited.get(s):
		        frontier.put( (COSTS[self.graph[start][s]], (s,[ self.graph[start][s] ],[s],COSTS[self.graph[start][s]]) ) )
		        infrontier[s] = True

		while not frontier.empty():
			# print(frontier)
			curr_node = frontier.get()
			#print(curr_node)
			curr_state = curr_node[1][0]
			curr_actionlist = curr_node[1][1]
			curr_vertexlist = curr_node[1][2]
			curr_cost = curr_node[1][3]

			visited[curr_state] = True
			infrontier[curr_state] = False

			if curr_state == goal:
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
			
		os.system('clear')
		
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

	def get_path(self,goal_row,goal_col,goal_dir,start_row=None,start_col=None,start_dir=None):
		if start_row == None:
			start_row = self.curr_row
		if start_col == None:
			start_col = self.curr_col
		if start_dir == None:
			start_dir = self.curr_dir
		if goal_dir == None:
			goal_dir = 'S'
		path = self.gridgraph.get_path( (start_row,start_col,start_dir), (goal_row,goal_col,goal_dir) )
		return path

	def update_pose(self,action,num_steps=1):
		action = action[:-1]+action[-1]*num_steps
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


	def set_curr_pose(self,r,c,dir):
		self.curr_row = r
		self.curr_col = c
		self.curr_dir = dir

	def __init__(self, dim, init_row=0, init_col=0, init_dir='N'):
		self.dim = dim
		self.cells = [[Cell(r,c) for c in range(dim)] for r in range(dim)]
		self.curr_row = init_row
		self.curr_col = init_col
		self.curr_dir = init_dir
		self.gridgraph = GridGraph(dim)


def get_optimized_path(actionlist,vertexlist,maze):
	num_actions = len(actionlist)
	if num_actions < 2:
		return actionlist, vertexlist, [1]*num_actions

	opt_actionlist = []
	opt_vertexlist = []
	opt_numlist = []

	for curr_action, curr_vertex in zip(actionlist,vertexlist):
		# concat only if last and current actions are the same AND last and current cell are visited
		if len(opt_actionlist) == 0:
			opt_actionlist.append(curr_action)
			opt_vertexlist.append(curr_vertex)
			opt_numlist.append(1)
		else:
			if curr_action == opt_actionlist[-1] and maze.is_visited(curr_vertex[0],curr_vertex[1]) and maze.is_visited(opt_vertexlist[-1][0],opt_vertexlist[-1][1]):
				opt_vertexlist[-1] = curr_vertex
				opt_numlist[-1] += 1
			else:
				opt_actionlist.append(curr_action)
				opt_vertexlist.append(curr_vertex)
				opt_numlist.append(1)

	# now add smooth turns
	idx = 1
	smooth_actionlist = []
	smooth_vertexlist = []
	smooth_numlist = []
	while len(opt_actionlist) > 0:

		curr_action = opt_actionlist.pop(0)
		curr_vertex = opt_vertexlist.pop(0)
		curr_num = opt_numlist.pop(0)

		#print(curr_action)
		#print(curr_vertex)
		#print(curr_num)
		#print(opt_actionlist)
		#print()

		if len(opt_actionlist) > 2:
			if curr_action == 'F' and (opt_actionlist[0] == 'R' or opt_actionlist[0] == 'L') and opt_actionlist[1] == 'F' and curr_num == 1 and maze.is_visited(curr_vertex[0],curr_vertex[1]) and maze.is_visited(opt_vertexlist[1][0],opt_vertexlist[1][1]):
				smooth_actionlist.append('F'+opt_actionlist[0]+'F')
				smooth_vertexlist.append(opt_vertexlist[1])
				smooth_numlist.append(opt_numlist[1])
				opt_actionlist.pop(0)
				opt_actionlist.pop(0)
				opt_vertexlist.pop(0)
				opt_vertexlist.pop(0)
				opt_numlist.pop(0)
				opt_numlist.pop(0)
			else:
				smooth_actionlist.append(curr_action)
				smooth_vertexlist.append(curr_vertex)
				smooth_numlist.append(curr_num)
		else:
			smooth_actionlist.append(curr_action)
			smooth_vertexlist.append(curr_vertex)
			smooth_numlist.append(curr_num)
	return smooth_actionlist,smooth_vertexlist,smooth_numlist
	#return opt_actionlist,opt_vertexlist,opt_numlist



def is_outlier(data,sample=None):
    # by default apply only to the last element of the data
    if sample==None:
        sample = data[-1]
    median = np.median(data)
    mad = np.median(np.abs(data-median))
    score = 0.6745*(sample-median)/mad
    return np.abs(score)>3.5



def odom_callback(msg):
	global pose
	pose = msg.pose.pose



def scan_callback(msg):
    global rangeshist, ranges, walls, wallstonowalls, nowallstowalls
    rangeshist[0].append(msg.ranges[0])
    rangeshist[1].append(msg.ranges[1])
    rangeshist[2].append(msg.ranges[2])

    for i in range(3):
    	#if not is_outlier(rangeshist[i],msg.ranges[i]):
		#robustranges[i] = msg.ranges[i]
	robustranges[i] = np.median(rangeshist[i])
    ranges = [msg.ranges[0],msg.ranges[1], msg.ranges[2]]

    prevwalls = walls
    walls = [ ranges[0]*np.sin(np.deg2rad(fwd_angle)) < MAX_SIDE_WALL_DIST, ranges[1] < MAX_FRONT_WALL_DIST, ranges[2]*np.sin(np.deg2rad(fwd_angle)) < MAX_SIDE_WALL_DIST ]
    wallstonowalls = [False, False, False]
    nowallstowalls = [False, False, False]

    for i in range(3):
        if not prevwalls[i] and walls[i]:
	    nowallstowalls[i] = True
	elif prevwalls[i] and not walls[i]:
	    wallstonowalls[i] = True


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
	while angle < num_steps*np.pi/2:

		quat = quaternion_multiply([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],prev_inv_orientation)
		angle += abs(euler_from_quaternion(quat)[2])
		prev_inv_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, -pose.orientation.w]

		asign = 0.0
		if angle > num_steps/2.0*np.pi/2:
			asign = -1.0
		elif angle < num_steps/2.0*np.pi/2:
			asign = 1.0
		else:
			asign = 0.0
		if abs(vel_msg.angular.z)+asign*WACCEL/RATE >= WMIN and abs(vel_msg.angular.z)+asign*WACCEL/RATE <= WMAX: 
			vel_msg.angular.z += sign*(asign*WACCEL)/RATE
		vel_pub.publish(vel_msg)
		# TODO fix the deceleration start time
		r.sleep()


def stop():
	vel_pub.publish(Twist())


def smoothturn(action,rwall=None,fwall=None,lwall=None):
	# this replaces a 1/2F R 1/2F or 1/2F L 1/2F combo
	
	vel_msg = Twist()
	if action=='L':
		sign = 1
	else:
		sign = -1

	radius = 0.05				# radius of the turn
	t = (np.pi*radius/2.0)/VMIN		# how much time to complete the turn

	straight_for_smoothturn( (LEN/2-radius+MOTOR_OFFSET), +1, rwall, fwall, lwall)
	r = rospy.Rate(RATE)
	prev_inv_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, -pose.orientation.w]
	angle = 0
	while angle < np.pi/2-np.pi/24:
		quat = quaternion_multiply([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],prev_inv_orientation)
		angle += abs(euler_from_quaternion(quat)[2])
		prev_inv_orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, -pose.orientation.w]

		vel_msg.linear.x = VMIN
		vel_msg.angular.z = sign*(np.pi/2)/t
		vel_pub.publish(vel_msg)
		r.sleep()	

	rwall, fwall, lwall = straight_for_smoothturn( (LEN/2-radius), +1)
	return rwall, fwall, lwall


def get_withincell_dist(dist):
	return (dist - np.floor(dist/LEN)*LEN)


def forward(num_steps=1, fwd_corr = True):
	vel_msg = Twist()
	vel_msg.linear.x = VMIN

	r = rospy.Rate(RATE) # hz
	dist = 0
	prev_err = 0.0
	prev_pos = pose.position

	lwall_hist = []
	fwall_hist = []
	rwall_hist = []

	while (robustranges[1] >= FWD_REF_DIST and dist <= num_steps*LEN) or dist <= LEN/4:

		# keep track of the distance traveled
		dist += np.sqrt((prev_pos.x - pose.position.x)*(prev_pos.x - pose.position.x) + (prev_pos.y - pose.position.y)*(prev_pos.y - pose.position.y))
		prev_pos = pose.position

		# set flags based on distance
		celldist = get_withincell_dist(dist)
		#print '{0:.3f}'.format(celldist)

		# correct distance when you expect to be close to a wall
		# if we round down, then get rid of the wall history, since we are starting afresh
		# if we round up, then adjust the distances stored 
		if (wallstonowalls[0] or wallstonowalls[2]) and (celldist < 2*LEN/10 or celldist > 8*LEN/10):
			if celldist < 2*LEN/10:
				print("Forward correction. Rounding down.")
				lwall_hist = []
				fwall_hist = []
				rwall_hist = []
			else:
				print("Forward correction. Rounding up.")
				print("WARN: need to implement distance wall history")
			print "Distance before: {0:0.2f}".format(dist)			
			dist = round(dist/LEN)*LEN
			print "Distance after: {0:0.2f}".format(dist)
			

		# decelerate when distance required to decelerate is more than dist left to travel
		# else accelerate as long as current speed is less than max
		asign = 0.0
		if (VMIN*VMIN-vel_msg.linear.x*vel_msg.linear.x)/2.0/(-LINACCEL) > num_steps*LEN-dist:
			asign = -1.0
		elif vel_msg.linear.x < VMAX:
			asign = +1.0
		else:
			asign = 0.0
		vel_msg.linear.x += (asign*LINACCEL)/RATE

		# most reliable wall information is between x% to y% distance traveled
		# this only works for the last step!
		if dist > (1*LEN/10 + (num_steps-1)*LEN) and dist < (4*LEN/10 + (num_steps-1)*LEN):
			rwall_hist.append(walls[0])
			fwall_hist.append(walls[1])
			lwall_hist.append(walls[2])

		# lateral correction
		err = 0
		if walls[0] and walls[1]:
			err = ranges[0]*np.sin(np.deg2rad(fwd_angle)) - ranges[2]*np.sin(np.deg2rad(fwd_angle))
		elif walls[0]:	# if right wall exists
			err = ranges[0]*np.sin(np.deg2rad(fwd_angle)) - (LEN/2)
		elif walls[2]:	# if left wall exists
			err = (LEN/2) - ranges[2]*np.sin(np.deg2rad(fwd_angle))
		
		print "Error: {0:.3f}".format(err)
		print "Angular velocity: {0:.2f}".format(vel_msg.angular.z)
		vel_msg.angular.z = err*KP + (err-prev_err)*KD
		prev_err = err
		vel_pub.publish(vel_msg)
		
		
		r.sleep()

	rwall= rwall_hist.count(True)>rwall_hist.count(False)
	fwall = fwall_hist.count(True)>fwall_hist.count(False)
	lwall = lwall_hist.count(True)>lwall_hist.count(False)
	
	# forward correction only when there is a front wall
	if fwall and fwd_corr:
		r = rospy.Rate(RATE) # hz
		timeout = RATE*1
		k_angcorr = 5
		k_lincorr = 2
		while (np.abs(ranges[0]-ranges[2]) > 0.001 or np.abs(ranges[1] - FWD_REF_DIST) > 0.001) and timeout > 0:
			vel_msg.linear.x =  (ranges[1] - FWD_REF_DIST)*k_lincorr
			vel_msg.angular.z = (ranges[0]-ranges[2])*k_angcorr
			vel_pub.publish(vel_msg)
			timeout -= 1
			r.sleep()
	
	return rwall, fwall, lwall


def halfforward(rwall=None,fwall=None,lwall=None):
	vel_msg = Twist()
	vel_msg.linear.x = VMIN

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

	while (robustranges[1] > FWD_REF_DIST and dist < LEN/2) or dist <= LEN/8:

		# keep track of the distance traveled
		dist += np.sqrt((prev_pos.x - pose.position.x)*(prev_pos.x - pose.position.x) + (prev_pos.y - pose.position.y)*(prev_pos.y - pose.position.y))
		prev_pos = pose.position

		# check for a wall to no wall transition
		# if you do, round off the distance traveled
		if (rwall == True and walls[0] == False) or (lwall == True and walls[2] == False):
			#print "Wall to no wall transition"
			#print "Before: {0:.2f}".format(dist)
			dist = round(dist/(LEN/2))*LEN/2
			#print "After: {0:.2f}".format(dist)
			break

		#print(rwall,lwall)
		#print(walls[0],walls[2])
		#print ""

		# check for no wall to wall transition
		# if you do, we will set the wall flag and wait for wall to no wall transition
		'''if (rwall == False and ranges[0]*np.sin(np.deg2rad(fwd_angle)) < LEN/2):
			#print "No wall to wall transition"
			rwall = True
			break
		if (lwall == False and ranges[2]*np.sin(np.deg2rad(fwd_angle)) < LEN/2):
			#print "No wall to wall transition"
			lwall = True
			break'''
		
		# record wall history	
		rwall_hist.append(walls[0])
		fwall_hist.append(walls[1])
		lwall_hist.append(walls[2])

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

		# lateral correction
		err = 0
		rcorr = (rwall == True or rwall_new == True) and (ranges[0]*np.sin(np.deg2rad(fwd_angle)) < MAX_SIDE_WALL_DIST)
		lcorr = (lwall == True or lwall_new == True) and (ranges[2]*np.sin(np.deg2rad(fwd_angle)) < MAX_SIDE_WALL_DIST)
		if lcorr and rcorr:
			err = (ranges[0]*np.sin(np.deg2rad(fwd_angle)) - ranges[2]*np.sin(np.deg2rad(fwd_angle)))/2
		elif rcorr:	# if right wall exists
			err = ranges[0]*np.sin(np.deg2rad(fwd_angle)) - (LEN/2)
		elif lcorr:	# if left wall exists
			err = (LEN/2) - ranges[2]*np.sin(np.deg2rad(fwd_angle))
		
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
	
	return rwall, fwall, lwall


def straight_for_smoothturn(targetdist,dir,rwall=None,fwall=None,lwall=None):

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

	while (dist <= targetdist and robustranges[1] >= FWD_REF_DIST+targetdist):
		# keep track of the distance traveled
		dist += np.sqrt((prev_pos.x - pose.position.x)*(prev_pos.x - pose.position.x) + (prev_pos.y - pose.position.y)*(prev_pos.y - pose.position.y))
		prev_pos = pose.position

		# record wall history
		rwall_hist.append(walls[0])
		fwall_hist.append(walls[1])
		lwall_hist.append(walls[2])

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

		# lateral correction
		err = 0
		rcorr = (rwall == True or rwall_new == True) and (ranges[0]*np.sin(np.deg2rad(fwd_angle)) < MAX_SIDE_WALL_DIST)
		lcorr = (lwall == True or lwall_new == True) and (ranges[2]*np.sin(np.deg2rad(fwd_angle)) < MAX_SIDE_WALL_DIST)
		if lcorr and rcorr:
			err = ranges[0]*np.sin(np.deg2rad(fwd_angle)) - ranges[2]*np.sin(np.deg2rad(fwd_angle))
		elif rcorr:	# if right wall exists
			err = ranges[0]*np.sin(np.deg2rad(fwd_angle)) - (LEN/2)
		elif lcorr:	# if left wall exists
			err = (LEN/2) - ranges[2]*np.sin(np.deg2rad(fwd_angle))
		
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
	while not walls[1] == False:
		turn('L',1)
		stop()
		action += 'L'

	action += 'F'
	r,f,l = halfforward()
	stop()
	maze.update_pose(action,1)
	maze.update_cell(r,f,l)
	return r,f,l


def execute_action(action,r=None,f=None,l=None):

	if action == 'F':
		halfforward(r,f,l)
		stop()
		r,f,l = halfforward()
	elif action == 'RF':
		r,f,l = smoothturn('R',r,f,l)
	elif action == 'LF':
		r,f,l = smoothturn('L',r,f,l)	
	elif action == 'RRF':
		halfforward(r,f,l)
		turn('R',2)
		r,f,l = halfforward()
	elif action == 'LLF':
		halfforward(r,f,l)
		turn('L',2)
		r,f,l = halfforward()
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
		r = None
		f = None
		l = None
	elif action == 'RR':
		halfforward(r,f,l)
		turn('R',2)
		r = None
		f = None
		l = None
	elif action == 'S':
		stop()
		r = None
		f = None
		l = None
	stop()
	return r,f,l


def wallfollow():
	r,f,l = halfforward()	
	stop()
	maze.update_pose('F',1)
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
		maze.update_pose(action,1)
		if not (r == None or f == None or l == None):
			maze.update_cell(r,f,l)
		maze.prettyprint_maze()


def goto(goal_row,goal_col):

	'''r,f,l = halfforward()	
	stop()
	maze.update_pose('F',1)
	maze.update_cell(r,f,l)'''
	
	r,f,l = make_first_move(maze)
	actionlist,vertexlist = maze.get_path(goal_row,goal_col,None)

	maze.prettyprint_maze(vertexlist)
	while len(actionlist) > 0:
		action = actionlist.pop(0)
		while not action[-1] == 'F' and len(actionlist) > 0:
			action += actionlist.pop(0)
		r,f,l = execute_action(action,r,f,l)
		maze.update_pose(action,1)
		if not (r == None or f == None or l == None):
			maze.update_cell(r,f,l)
		actionlist,vertexlist = maze.get_path(goal_row,goal_col,None)
		maze.prettyprint_maze(vertexlist)
	if action[-1] == 'F':
		halfforward(r,f,l)

def test():

	# Starts a new node
	rospy.init_node('vayu', anonymous=False)
	rospy.Subscriber('scan', LaserScan, scan_callback, queue_size=1)
	rospy.Subscriber('odom', Odometry, odom_callback, queue_size=1)
	
	# wait until you get first scan
	r = rospy.Rate(RATE)
	while len(ranges) < 3:
		print("Waiting for scan")
		print(ranges)
		r.sleep()

	maze = Maze(16,0,0)
	goal_row = 8
	goal_col = 8
	
	r,f,l = make_first_move(maze)
	print(r,f,l)
			

def init():
    global ranges, maze

    # Starts a new node
    rospy.init_node('vayu', anonymous=False)
    rospy.Subscriber('/scan', LaserScan, scan_callback, queue_size=1)
    rospy.Subscriber('/odom', Odometry, odom_callback, queue_size=1)
    # wait until you get first scan
    r = rospy.Rate(RATE)
    while len(ranges) < 3:
        print("Waiting for scan")
	print(ranges)
	r.sleep()

    maze = Maze(16,0,0)

    #wallfollow()
    goto(8,8)
    rospy.sleep(2)
    goto(0,0)
    rospy.sleep(2)
    goto(8,8)
    rospy.sleep(2)
    goto(0,0)
    rospy.sleep(2)
    goto(8,8)
    rospy.sleep(2)
    goto(0,0)


if __name__ == '__main__':
    try:
        #Testing our function
	#test()
        init()
    except rospy.ROSInterruptException: pass
