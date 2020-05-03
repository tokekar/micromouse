#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
import queue, collections
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

linspeed = 0.05
angspeed = np.pi/2
LEN = 0.18
vel_pub = rospy.Publisher('/kbot/base_controller/cmd_vel', Twist, queue_size=10)
kp = -4
fwd_angle = 60

rangeshist = [collections.deque(maxlen=10), collections.deque(maxlen=10), collections.deque(maxlen=1)]
robustranges = [0,0,0]
ranges = []
walls = []
pose = []

DIRS = {'N':0,'W':1,'S':2,'E':3, 0:'N', 1:'W', 2:'S', 3:'E'} # bijective mapping to keep it easier to do modular math
ACTIONS = ['L','F','R','S']
COSTS = {'F': 1, 'L': 5, 'R': 5}

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

			if curr_state[0] == goal[0] and curr_state[1] == goal[1]:
				#print("Found solution:", curr_path)
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
		if action ==  'F':
			if self.curr_dir == 'N':
				self.set_curr_pose(self.curr_row+num_steps,self.curr_col,self.curr_dir)
			elif self.curr_dir == 'S':
				self.set_curr_pose(self.curr_row-num_steps,self.curr_col,self.curr_dir)
			elif self.curr_dir == 'E':
				self.set_curr_pose(self.curr_row,self.curr_col+num_steps,self.curr_dir)
			elif self.curr_dir == 'W':
				self.set_curr_pose(self.curr_row,self.curr_col-num_steps,self.curr_dir)
		elif action == 'R':
			self.set_curr_pose(self.curr_row,self.curr_col,DIRS[np.mod(DIRS[self.curr_dir]-num_steps,4)])
		elif action == 'L':
			self.set_curr_pose(self.curr_row,self.curr_col,DIRS[np.mod(DIRS[self.curr_dir]+num_steps,4)])
		elif action == 'S':
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
	return opt_actionlist,opt_vertexlist,opt_numlist



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
    # TODO this is a hack, use proper indexing
    global rangeshist, ranges, walls

    rangeshist[0].append(msg.ranges[0])
    rangeshist[1].append(msg.ranges[1])
    rangeshist[2].append(msg.ranges[2])

    for i in range(3):
    	if not is_outlier(rangeshist[i],msg.ranges[i]):
		#robustranges[i] = msg.ranges[i]
		robustranges[i] = np.median(rangeshist[i])
    ranges = [msg.ranges[0],msg.ranges[1], msg.ranges[2]]
    walls = [ ranges[0]*np.sin(np.deg2rad(fwd_angle)) < LEN, ranges[1]<LEN, ranges[2]*np.sin(np.deg2rad(fwd_angle)) < LEN ]


def turn(action,num_steps=1):
	vel_msg = Twist()

	if action=='L':
		sign = 1
	else:
		sign = -1
	vel_msg.linear.x = 0
	vel_msg.angular.z = sign*angspeed
	r = rospy.Rate(10) # 10hz

	for i in range( int(num_steps*(np.pi/2)/angspeed*10) ):
		#Publish the velocity
		vel_pub.publish(vel_msg)
		r.sleep()


def stop():
	vel_pub.publish(Twist())


def forward(num_steps=1):
	vel_msg = Twist()
	vel_msg.linear.x = linspeed*min(num_steps,2)

	r = rospy.Rate(20) # hz
	dist = 0
	prev_pos = pose.position

	lwall_hist = []
	fwall_hist = []
	rwall_hist = []

	while (robustranges[1] > LEN/2-0.035 and dist < num_steps*LEN) or (dist < LEN/4):

		# keep track of the distance traveled
		dist += np.sqrt((prev_pos.x - pose.position.x)*(prev_pos.x - pose.position.x) + (prev_pos.y - pose.position.y)*(prev_pos.y - pose.position.y))
		prev_pos = pose.position

		# most reliable wall information is between 40% to 60% distnace traveled
		# this only works when you travel one step!
		if dist > 2*LEN/5 and dist < 3*LEN/5:
			rwall_hist.append(walls[0])
			fwall_hist.append(walls[1])
			lwall_hist.append(walls[2])

		err = []
		if walls[0]:	# if right wall exists
			err.append(ranges[0] - (LEN/2)/np.sin(np.deg2rad(fwd_angle)))
		if walls[2]:	# if left wall exists
			err.append((LEN/2)/np.sin(np.deg2rad(fwd_angle)) - ranges[2])
		if len(err) > 0:
			err = np.mean(err)
		else:
			err = 0
		vel_msg.angular.z = err*kp
		vel_pub.publish(vel_msg)
		r.sleep()

	return rwall_hist.count(True)>rwall_hist.count(False), fwall_hist.count(True)>fwall_hist.count(False), lwall_hist.count(True)>lwall_hist.count(False)


def execute_action(action,num_steps):
	r = None
	f = None
	l = None

	if action == 'F':
		r,f,l = forward(num_steps)
		stop()
	elif action == 'R' or action == 'L':
		turn(action,num_steps)
		stop()
	elif action == 'S':
		stop()
	return r,f,l


def goto(goal_row,goal_col):
	# Draw maze
	plt.axis([0, maze.dim+1, 0, maze.dim+1])
	actionlist,vertexlist = maze.get_path(goal_row,goal_col,None)
	actionlist,_,numlist = get_optimized_path(actionlist,vertexlist,maze)

	while len(actionlist) > 0:
		action = actionlist.pop(0)
		num_steps = numlist.pop(0)
		r,f,l = execute_action(action,num_steps)
		maze.update_pose(action,num_steps)
		print(maze.curr_row, maze.curr_col, maze.curr_dir)

		if action == 'F' and num_steps == 1:
			maze.update_cell(r,f,l)
			actionlist,vertexlist = maze.get_path(goal_row,goal_col,None)
			actionlist,_,numlist = get_optimized_path(actionlist,vertexlist,maze)

		#maze.print_maze()
		maze.draw_maze()


def test():

	# Starts a new node
	rospy.init_node('vayu', anonymous=False)
	rospy.Subscriber('/kbot/laser_scan/scan', LaserScan, scan_callback, queue_size=1)
	rospy.Subscriber('/kbot/base_controller/odom', Odometry, odom_callback, queue_size=1)
    # wait until you get first scan
	r = rospy.Rate(10)
	while len(ranges) < 3:
		print("Waiting for scan")
		print(ranges)
		r.sleep()

	maze = Maze(4,0,0)
	maze.update_cell(True,False,True,0,0,'N')
	maze.update_cell(True,False,True,1,0,'N')
	maze.update_cell(True,False,True,2,0,'N')
	maze.update_cell(False,False,True,3,0,'N')

	actionlist,vertexlist = maze.get_path(3,3,'N')
	print(actionlist)
	print(vertexlist)
	actionlist,vertexlist,numlist = get_optimized_path(actionlist,vertexlist,maze)
	for a,n,v in zip(actionlist,numlist,vertexlist):
		print(a,n,v)
		execute_action(a,n)

def init():
    global ranges, maze

    # Starts a new node
    rospy.init_node('vayu', anonymous=False)
    rospy.Subscriber('/kbot/laser_scan/scan', LaserScan, scan_callback, queue_size=1)
    rospy.Subscriber('/kbot/base_controller/odom', Odometry, odom_callback, queue_size=1)
    # wait until you get first scan
    r = rospy.Rate(10)
    while len(ranges) < 3:
        print("Waiting for scan")
	print(ranges)
	r.sleep()

    maze = Maze(8,0,0)

    #wallfollow()
    goto(3,4)
    goto(0,0)
    goto(3,4)
    goto(0,0)


if __name__ == '__main__':
    try:
        #Testing our function
		#test()
        init()
    except rospy.ROSInterruptException: pass
