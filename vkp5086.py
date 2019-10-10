############################################################
# CMPSC 442: Homework 3
############################################################

student_name = "Vennila Pugazhenthi"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import math
from collections import deque
import random
import copy
from queue import PriorityQueue
import sys

############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    #new_tile=TilePuzzle([[0 for x in range(cols)] for y in range(rows)])
    new_tile=[[0 for x in range(cols)]for y in range(rows)]
    count=0
    for i in range(rows):
        for j in range(cols):
            if(not ((i==rows-1)and(j==cols-1))):
                count=count+1
                new_tile[i][j]=count
            #if((i==rows-1)and(j==cols-1)):
                #print("Here at:")
                #print(i,j)
            #else:
            #    new_tile[i][j]=0

    return TilePuzzle(new_tile)


class TilePuzzle(object):
    
    # Required
    def __init__(self, board):
        self.board=board
        self.row=len(board)
        self.col=len(board[0])
        for i in range(self.row):
            for j in range(self.col):
                if(self.board[i][j]==0):
                    self.empty=[i,j]
        #self.empty=[self.row-1,self.col-1]

    def get_board(self):
        return self.board

    def perform_move(self, direction):
        moved=False
        if(direction=="up"):
            if((self.empty[0]-1 < 0)or (self.empty[0]-1>=self.row)):
                return moved
            else:
                old_row=self.empty[0]
                old_col=self.empty[1]
                new_row=self.empty[0]-1
                new_col=self.empty[1]
                self.empty[0]=self.empty[0]-1
                temp=self.board[new_row][new_col]
                self.board[new_row][new_col]=0
                self.board[old_row][old_col]=temp
                moved=True
                return moved
        if(direction=="down"):
            if((self.empty[0]+1<0)or(self.empty[0]+1>=self.row)):
                return moved
            else:
                old_row=self.empty[0]
                old_col=self.empty[1]
                new_row=self.empty[0]+1
                new_col=self.empty[1]
                self.empty[0]=self.empty[0]+1
                temp=self.board[new_row][new_col]
                self.board[new_row][new_col]=0
                self.board[old_row][old_col]=temp
                moved=True
                return moved
        if(direction=="left"):
            if((self.empty[1]-1<0)or(self.empty[1]-1>=self.col)):
                return moved
            else:
                old_row=self.empty[0]
                old_col=self.empty[1]
                new_row=self.empty[0]
                new_col=self.empty[1]-1
                self.empty[1]=self.empty[1]-1
                temp=self.board[new_row][new_col]
                self.board[new_row][new_col]=0
                self.board[old_row][old_col]=temp
                moved=True
                return moved
        if(direction=="right"):
            if((self.empty[1]+1<0)or(self.empty[1]+1>=self.col)):
                return moved
            else:
                old_row=self.empty[0]
                old_col=self.empty[1]
                new_row=self.empty[0]
                new_col=self.empty[1]+1
                self.empty[1]=self.empty[1]+1
                temp=self.board[new_row][new_col]
                self.board[new_row][new_col]=0
                self.board[old_row][old_col]=temp
                moved=True
                return moved


    def scramble(self, num_moves):
        direction=["up","down","left","right"]
        for i in range(num_moves):
            pick=random.choice(direction)
            self.perform_move(pick)
        #return self.board


    def is_solved(self):
        start=[[0 for x in range(self.col)] for y in range(self.row)]
        count=0
        for i in range(self.row):
            for j in range(self.col):
                if((i==self.row-1)and(j==self.col-1)):
                    start[i][j]=0
                else:
                    count=count+1
                    start[i][j]=count
        if(self.board==start):
            return True
        else:
            return False


    def copy(self):
        #temp=copy.deepcopy(self.board)
        temp=copy.deepcopy(self)
        return temp

    def successors(self):
        direction=["up","down","right","left"]

        for x in range(len(direction)):
            temp=self.copy()
            check=temp.perform_move(direction[x])
            if check==True:
                yield direction[x],temp


    # Required
    def find_solutions_iddfs(self):
        is_solved=False
        count=1
        while not is_solved:
            for sol in self.iddfs_helper(count,[]):
                if sol is not None:
                    yield sol
                    if(self.is_solved()):
                        is_solved=True
            count=count+1

    def iddfs_helper(self,limit,moves):
        if limit==0:
            if self.is_solved():
                yield moves
            else:
                yield None
        else:
            for dir,new in self.successors():
                for sol in new.iddfs_helper(limit-1,moves+[dir]):
                    yield sol

    # Required
    def find_solution_a_star(self):
        PQ=PriorityQueue()
        f=0.0
        g=0.0
        h=0.0
        parent=[]
        #path=list()
        track=[]
        cellDetails=[self,g,h,parent]
        thisdict={f:cellDetails}
        #element=(f,cellDetails)
        PQ.put(f)
        while not PQ.empty():
            f_old=PQ.get()
            #print(f_old)
            cellDetails_old=thisdict.get(f_old)
            old=cellDetails_old[0]
            g_old=cellDetails_old[1]
            h_old=cellDetails_old[2]
            parent_old=cellDetails_old[3]
            track.append(old.get_board())
            if(old.is_solved()):
                return parent_old
            for path,next in old.successors():
                    if next.get_board() not in track:
                        g_new = g_old + 1
                        h_new = next.getManhattanDistance()
                        f_new = g_new + h_new
                        new = next
                        parent_new = parent_old + [path]
                        cellDetails_new = [new, g_new, h_new, parent_new]
                        thisdict[f_new] = cellDetails_new
                        PQ.put(f_new)



        return


    def getManhattanDistance(self):
        h=0.0
        for i in range(self.row):
            for j in range(self.col):
                value=self.board[i][j]
                if(value!=0):
                    goal_x=(value-1)/self.row
                    goal_y=(value-1)%self.row
                    h+=abs(i-goal_x)+abs(j-goal_y)
        return h
    # def getManhattanDistance(self):
    #     md=0
    #     for r in range(self.row):
    #         for c in range(self.col):
    #             current=self.board[r][c]
    #             position=self.goal_state(current)
    #             goal_x=position[0]
    #             goal_y=position[1]
    #             current_x=r
    #             current_y=c
    #             md+=abs(goal_x-current_x)+abs(goal_y-current_y)
    #     return md

    # def goal_state(self,current):
    #     start = [[0 for x in range(self.col)] for y in range(self.row)]
    #     count = 0
    #     for i in range(self.row):
    #         for j in range(self.col):
    #             if ((i == self.row - 1) and (j == self.col - 1)):
    #                 start[i][j] = 0
    #             else:
    #                 count = count + 1
    #                 start[i][j] = count
    #     for i in range(self.row):
    #         for j in range(self.col):
    #             if(start[i][j]==current):
    #                 return [i,j]

#p=create_tile_puzzle(3,3)
#b=[[1,2,3],[4,0,5],[6,7,8]]
#p=TilePuzzle(b)
#for move,p_new in p.successors():
 #   print(move,p_new.get_board())
#b=[[4,1,2],[0,5,3],[7,8,6]]
#p=TilePuzzle(b)
#solution = p.find_solutions_iddfs()
#print(next(solution))
#b=create_tile_puzzle(3,3)
#b.perform_move("up")
#print(b.getManhattanDistance())
#b=[[4,1,2],[0,5,3],[7,8,6]]
#p=TilePuzzle(b)
#print(list(p.find_solutions_iddfs()))
#print(next(solution))
############################################################
# Section 2: Grid Navigation
############################################################

# A class that holds necessary parameters
class Node():
    def __init__(self,parent=None):   #def __init__(self,parent=None,position=None):
        self.parent = parent
        #self.position = position

        self.g = math.inf
        self.h = math.inf
        self.f = math.inf

# A Utility Function to check whether given node is valid or not
def isValid(node,scene):
    row=len(scene)
    col=len(scene[0])
    return (node[0]>=0)and(node[1]>=0)and(node[0]<=row-1)and(node[1]<=col-1)

# Calculates the Euclidean distance
def calculateHValue(node,goal):
    x1= node[0]
    x2=goal[0]
    y1=node[1]
    y2=goal[1]
    difference_x= x1-x2
    difference_y=y1-y2
    square_x= difference_x * difference_x
    square_y= difference_y*difference_y
    H=math.sqrt(square_x+square_y)
    return H

# This function check if the given node is contains block
# Returns True if it is blocked else False
def CheckUnBlocked(node,scene):
    x=node[0]
    y=node[1]
    if scene[x][y]==True:
        return False
    else:
        return True

# This function checks whether goal cell has been reached or not
# Returns True if its reached else false
def IsDestination(node,goal):
    if((node[0]==goal[0])and (node[1]==goal[1])):
        return True
    else:
        return False

def tracePath(cellDetails,goal):
    #print ("\nThe path is ")
    row=goal[0]
    col=goal[1]
    path = deque()
    while(not(cellDetails[row][col].parent==(row,col))):
        path.append((row,col))
        temp=cellDetails[row][col].parent
        row=temp[0]
        col=temp[1]
    path.append((row,col))
    temp2=[]
    while(len(path)!=0):
        p=path.pop()
        temp2.append(p)
        #print ("-> ",p)

    #return temp2
    return temp2
    #return



# A Function to find the shortest path between start and goal cell according to A* search
def find_path(start, goal, scene):
    # If the start is out of range
    if(isValid(start,scene)==False):
        return None
    # If the goal is out of range
    if(isValid(goal,scene)==False):
        return None
    # Either source or goal is blocked
    if((CheckUnBlocked(start,scene)==False)or (CheckUnBlocked(goal,scene)==False)):
        return None
    # If the goal and start cell are the same
    if(IsDestination(start,goal)==True):
        return list(start)

    row = len(scene)
    col = len(scene[0])
    # Closed List initialised with False which means that no cells has been included yet
    closedList=[[False for x in range(col)] for y in range (row)]

    # 2D array to hold the details of the cell
    cellDetails=[[Node() for x in range(col)] for y in range(row)]

    # Initialising the parameter of the starting node
    x=start[0]
    y=start[1]
    cellDetails[x][y].f=0.0
    cellDetails[x][y].g=0.0
    cellDetails[x][y].h=0.0
    cellDetails[x][y].parent=start

    # Create a set <f,<i,j>> where f=g+h
    openList=set()

    # Insert the starting cell
    openList.add((0.0,(x,y)))

    #Boolean value as false as the initially the destination is not reached
    foundDest=False

    while(len(openList)!=0):
        p=openList.pop()

        # Add this vertex to the closed List
        i=p[1][0]
        j=p[1][1]
        closedList[i][j]=True

        # Generating all the 8 successor of this cell
        # Cell-> Popped Cell (i,j)
        # Up-> (i-1,j)
        # Down-> (i+1,j)
        # Right-> (i,j+1)
        # Left-> (i,j-1)
        # Up-right-> (i-1,j+1)
        # Up-left-> (i-1,j-1)
        # Down-right-> (i+1,j+1)
        # Down-left-> (i+1,j-1)

        # To store new g,h,f
        gNew=0.0
        hNew=0.0
        fNew=0.0

        # 1st Successor (Up)
        if(isValid((i-1,j),scene)==True):
            if(IsDestination((i-1,j),goal)==True):
                cellDetails[i-1][j].parent=(i,j)
                #print("The destination cell is found\n")
                return tracePath(cellDetails,goal)
                foundDest=True
                return
            elif ((closedList[i-1][j]==False) and (CheckUnBlocked((i-1,j),scene)==True)):
                gNew=cellDetails[i][j].g+1.0
                hNew=calculateHValue((i-1,j),goal)
                fNew= gNew+hNew
                if(cellDetails[i-1][j].f==math.inf or cellDetails[i-1][j].f>fNew):
                    openList.add((fNew,(i-1,j)))
                    cellDetails[i-1][j].f=fNew
                    cellDetails[i-1][j].g=gNew
                    cellDetails[i-1][j].h=hNew
                    cellDetails[i-1][j].parent=(i,j)

        # 2nd Successor (Down)
        if (isValid((i + 1, j), scene) == True):
            if (IsDestination((i + 1, j), goal) == True):
                cellDetails[i + 1][j].parent = (i, j)
                #print("The destination cell is found\n")
                return tracePath(cellDetails, goal)
                foundDest = True
                return
            elif ((closedList[i + 1][j] == False) and (CheckUnBlocked((i + 1, j), scene) == True)):
                gNew = cellDetails[i][j].g + 1.0
                hNew = calculateHValue((i + 1, j), goal)
                fNew = gNew + hNew
                if (cellDetails[i + 1][j].f == math.inf or cellDetails[i + 1][j].f > fNew):
                    openList.add((fNew, (i + 1, j)))
                    cellDetails[i + 1][j].f = fNew
                    cellDetails[i + 1][j].g = gNew
                    cellDetails[i + 1][j].h = hNew
                    cellDetails[i + 1][j].parent = (i, j)

        # 3rd Successor (Right)
        if (isValid((i , j+1), scene) == True):
            if (IsDestination((i , j+1), goal) == True):
                cellDetails[i][j+1].parent = (i, j)
                #print("The destination cell is found\n")
                return tracePath(cellDetails, goal)
                foundDest = True
                return
            elif ((closedList[i][j+1] == False) and (CheckUnBlocked((i, j+1), scene) == True)):
                gNew = cellDetails[i][j].g + 1.0
                hNew = calculateHValue((i, j+1), goal)
                fNew = gNew + hNew
                if (cellDetails[i][j+1].f == math.inf or cellDetails[i][j+1].f > fNew):
                    openList.add((fNew, (i , j+1)))
                    cellDetails[i][j+1].f = fNew
                    cellDetails[i][j+1].g = gNew
                    cellDetails[i][j+1].h = hNew
                    cellDetails[i][j+1].parent = (i, j)
        # 4th successor (Left)
        if (isValid((i , j-1), scene) == True):
            if (IsDestination((i , j-1), goal) == True):
                cellDetails[i][j-1].parent = (i, j)
                #print("The destination cell is found\n")
                return tracePath(cellDetails, goal)
                foundDest = True
                return
            elif ((closedList[i][j-1] == False) and (CheckUnBlocked((i, j-1), scene) == True)):
                gNew = cellDetails[i][j].g + 1.0
                hNew = calculateHValue((i, j-1), goal)
                fNew = gNew + hNew
                if (cellDetails[i][j-1].f == math.inf or cellDetails[i][j-1].f > fNew):
                    openList.add((fNew, (i , j-1)))
                    cellDetails[i][j-1].f = fNew
                    cellDetails[i][j-1].g = gNew
                    cellDetails[i][j-1].h = hNew
                    cellDetails[i][j-1].parent = (i, j)

        # 5th Successor (Up-Right)
        if (isValid((i-1 , j+1), scene) == True):
            if (IsDestination((i-1 , j+1), goal) == True):
                cellDetails[i-1][j+1].parent = (i, j)
                #print("The destination cell is found\n")
                return tracePath(cellDetails, goal)
                foundDest = True
                return
            elif ((closedList[i-1][j+1] == False) and (CheckUnBlocked((i-1, j+1), scene) == True)):
                gNew = cellDetails[i][j].g + 1.0
                hNew = calculateHValue((i-1, j+1), goal)
                fNew = gNew + hNew
                if (cellDetails[i-1][j+1].f == math.inf or cellDetails[i-1][j+1].f > fNew):
                    openList.add((fNew, (i-1 , j+1)))
                    cellDetails[i-1][j+1].f = fNew
                    cellDetails[i-1][j+1].g = gNew
                    cellDetails[i-1][j+1].h = hNew
                    cellDetails[i-1][j+1].parent = (i, j)

        # 6th Successor (Up-Left)
        if (isValid((i-1 , j-1), scene) == True):
            if (IsDestination((i-1 , j-1), goal) == True):
                cellDetails[i-1][j-1].parent = (i, j)
                #print("The destination cell is found\n")
                return tracePath(cellDetails, goal)
                foundDest = True
                return
            elif ((closedList[i-1][j-1] == False) and (CheckUnBlocked((i-1, j-1), scene) == True)):
                gNew = cellDetails[i][j].g + 1.0
                hNew = calculateHValue((i-1, j-1), goal)
                fNew = gNew + hNew
                if (cellDetails[i-1][j-1].f == math.inf or cellDetails[i-1][j-1].f > fNew):
                    openList.add((fNew, (i-1 , j-1)))
                    cellDetails[i-1][j-1].f = fNew
                    cellDetails[i-1][j-1].g = gNew
                    cellDetails[i-1][j-1].h = hNew
                    cellDetails[i-1][j-1].parent = (i, j)

        # 7th Successor (Down-Right)
        if (isValid((i + 1, j + 1), scene) == True):
            if (IsDestination((i + 1, j + 1), goal) == True):
                cellDetails[i + 1][j + 1].parent = (i, j)
                #print("The destination cell is found\n")
                return tracePath(cellDetails, goal)
                foundDest = True
                return
            elif ((closedList[i + 1][j + 1] == False) and (CheckUnBlocked((i + 1, j + 1), scene) == True)):
                gNew = cellDetails[i][j].g + 1.0
                hNew = calculateHValue((i + 1, j + 1), goal)
                fNew = gNew + hNew
                if (cellDetails[i + 1][j + 1].f == math.inf or cellDetails[i + 1][j + 1].f > fNew):
                    openList.add((fNew, (i + 1, j + 1)))
                    cellDetails[i + 1][j + 1].f = fNew
                    cellDetails[i + 1][j + 1].g = gNew
                    cellDetails[i + 1][j + 1].h = hNew
                    cellDetails[i + 1][j + 1].parent = (i, j)

        # 8th Successor (Down-Left)
        if (isValid((i + 1, j - 1), scene) == True):
            if (IsDestination((i + 1, j - 1), goal) == True):
                cellDetails[i + 1][j - 1].parent = (i, j)
                #print("The destination cell is found\n")
                return tracePath(cellDetails, goal)
                foundDest = True
                return
            elif ((closedList[i + 1][j - 1] == False) and (CheckUnBlocked((i + 1, j - 1), scene) == True)):
                gNew = cellDetails[i][j].g + 1.0
                hNew = calculateHValue((i + 1, j - 1), goal)
                fNew = gNew + hNew
                if (cellDetails[i + 1][j - 1].f == math.inf or cellDetails[i + 1][j - 1].f > fNew):
                    openList.add((fNew, (i + 1, j - 1)))
                    cellDetails[i + 1][j - 1].f = fNew
                    cellDetails[i + 1][j - 1].g = gNew
                    cellDetails[i + 1][j - 1].h = hNew
                    cellDetails[i + 1][j - 1].parent = (i, j)
    #if(foundDest==False):
        #print ("Failed to find the Destination Cell\n")

    return


#scene=[[False,False,False],[False,True,False],[False,False,False]]
#print(find_path((0,0),(2,1),scene))
#find_path((0,0),(2,1),scene)
#python homework3_grid_navigation_gui.py scene


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################

def solve_distinct_disks(length, n):
    pass

############################################################
# Section 4: Dominoes Game
############################################################

def create_dominoes_game(rows, cols):
    new_dominoes=[[False for x in range(cols)]for y in range(rows)]
    return DominoesGame(new_dominoes)

class DominoesGame(object):

    # Required
    def __init__(self, board):
        self.board=board
        self.row=len(board)
        self.col=len(board[0])
        #self.leaf_count=0

    def get_board(self):
        return self.board

    def reset(self):
        for x in range(self.row):
            for y in range(self.col):
                self.board[x][y]=False

    def is_legal_move(self, row, col, vertical):
        if (vertical==True):
            if((row+1>=0)and(row+1<self.row)and(col>=0)and(col<self.col)):
                if((self.board[row][col]==False)and(self.board[row+1][col]==False)):
                    return True
        if(vertical==False):
            if((row>=0)and(row<self.row)and(col+1>=0)and(col+1<self.col)):
                if((self.board[row][col]==False)and(self.board[row][col+1]==False)):
                    return True
        return False

    def legal_moves(self, vertical):
        #possibilities=[]
            for i in range(self.row):
                for j in range(self.col):
                    if(self.is_legal_move(i,j,vertical)):
                        yield (i,j)


    def perform_move(self, row, col, vertical):
        if(self.is_legal_move(row,col,vertical)):
            if(vertical==True):
                self.board[row][col]=True
                self.board[row+1][col]=True
            if(vertical==False):
                self.board[row][col]=True
                self.board[row][col+1]=True

    def game_over(self, vertical):
        b=list(self.legal_moves(vertical))
        if(b ==True):
            return True
        else:
            return False


    def copy(self):
        temp = copy.deepcopy(self)
        return temp

    def successors(self, vertical):
        for next_move in self.legal_moves(vertical):
            new_copy=self.copy()
            new_copy.perform_move(next_move[0],next_move[1],vertical)
            yield (next_move,new_copy)

    def get_random_move(self, vertical):
        moves=list(self.legal_moves(vertical))
        return random.choice(moves)

    def evaluate_score(self, vertical):
         length1=len(list(self.legal_moves(vertical)))
         length2=len(list(self.legal_moves(not vertical)))
         return [length1,length2]

    def get_best_move(self, vertical, limit):
        return self.max_value(None,vertical,limit,-math.inf,math.inf)

    def max_value(self,node,vertical,limit,alpha,beta):
        if limit == 0 or self.game_over(vertical):
            num=self.evaluate_score(vertical)
            difference=num[0]-num[1]
            return node, difference, 1

        move=node
        v = -math.inf
        count = 0

        for next, new_node in self.successors(vertical):
            new_move,new_score,new_count = new_node.min_value(next,not vertical,limit-1,alpha,beta)
            count += new_count
            if new_score > v:
                v = new_score
                move = next
            if v >= beta:
                return move, v, count
            alpha = max(alpha, v)

        return move, v, count

    def min_value(self,node,vertical,limit,alpha,beta):

        if limit == 0 or self.game_over(vertical):
            num=self.evaluate_score(vertical)
            difference=num[1]-num[0]
            return node, difference, 1

        move= node
        v = math.inf
        count = 0
        for next, new_node in self.successors(vertical):
            new_move, new_score, new_count = new_node.max_value(next,not vertical,limit-1,alpha,beta)
            count += new_count
            if new_score < v:
                v = new_score
                move = next
            if v <= alpha:
                return move, v, count
            beta = min(beta, v)

        return move, v, count

#b=[[False,False,False],[False,False,False],[False,False,False]]
#g=DominoesGame(b)
#print(g.get_best_move(True,1))
#print(g.get_best_move(True,2))
#b2=[[False,False,False],[False,False,False],[False,False,False]]
#g2=DominoesGame(b2)
#g2.perform_move(0,1,True)
#print(g2.get_best_move(False,1))
#print(g2.get_best_move(False,1))
#print(g2.get_best_move(False,2))


############################################################
# Section 5: Feedback
############################################################

feedback_question_1 = """
I spent at least 35-40 hrs on this homework.
"""

feedback_question_2 = """
The most challenging part was Tile Puzzle where we had to solve in both the algorithms.
I stumbled in implementing it in the iddfs algorithm. 
"""

feedback_question_3 = """
I liked working on the grid navigation problem. I would like the question 2 & 3 to be broken 
into parts like other questions.
"""
