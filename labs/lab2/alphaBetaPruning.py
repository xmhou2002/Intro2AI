#################################################################
# 棋盘位置表示（0-8）:                                          #
# 0  1  2                                                       #
# 3  4  5                                                       #
# 6  7  8                                                       #
#################################################################
import copy

MARKERS = ['_', 'O', 'X']
Human_token = -1 # 由人类对手下
PEOPLE = -1

AI = 1
AI_token = 1 # 永远由AI下

Open_token = 0

# 设定获胜的组合方式(横、竖、斜)
WINNING_TRIADS = ((0, 1, 2), (3, 4, 5), (6, 7, 8),
                  (0, 3, 6), (1, 4, 7),(2, 5, 8),
                  (0, 4, 8), (2, 4, 6))
# 设定棋盘按一行三个打印
PRINTING_TRIADS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
# 用一维列表表示棋盘:
SLOTS = (0, 1, 2, 3, 4, 5, 6, 7, 8)


def print_board(board):
    """打印当前棋盘"""
    for row in PRINTING_TRIADS:
        r = ' '
        for hole in row:
            r += MARKERS[board[hole]] + ' '
        print(r)

def legal_move_left(board):
    """ 判断棋盘上是否还有空位 """
    for i in SLOTS:
        if board[i] == Open_token:
            return True
    return False

def winner(board):
    """ 判断局面的胜者,返回值-1表示X获胜,1表示O获胜,0表示平局或者未结束"""
    for triad in WINNING_TRIADS:
        triad_sum = board[triad[0]] + board[triad[1]] + board[triad[2]]
        if triad_sum == 3 or triad_sum == -3:
            return board[triad[0]]  # 表示棋子的数值恰好也是-1:X,1:O
    return 0

#拓展函数
def next(board, side):
    sons = []
    for i in SLOTS: # SLOTS is the listzation of selection space
        if board[i] == Open_token:
            son=copy.deepcopy(board)
            son[i]=side # side meets the token of this side
            sons.append([son, i])
    return sons

DEPTH = 4
CENTRALSLOT = 4
#搜索函数
# as a recursive function, transfer depth, alpha and beta recursively
# return evaluation value and its SLOT choice
def move(board, depth=0, alpha=float('-inf'), beta=float('inf') ):
    # if AI first, return (arbitrary value and) centralSLOT directly
    # if depth==1 or depth==0:
    #     a=1
    if board==[] or board==[0, 0, 0, 0, 0, 0, 0, 0, 0]:
        return 0, CENTRALSLOT
    # if one side wins, return infinity (and arbitrary SLOT)
    if winner(board) == Human_token:
        return float('-inf') , 0
    if winner(board) == AI_token:
        return float('inf') , 0
    # judge the side to expand by depth
    side = AI_token
    if depth%2==1:
        side = Human_token
    
    # AI side: expand and pick the maximum of son node values as its value
    if side == AI_token:
        # if this node is a leaf node, estimate value directly, return this value (and arbitrary SLOT)
        if depth==DEPTH:
            v =evaluate(board)
            return v, 0
        # if this node is the full board of dogfall, return arbitrary value and arbitrary SLOT
        if legal_move_left(board)==False:
            return 0, 0

        # not leaf node: traverse son nodes, calculate the value recursively
        # hAlpha records the current maximum, mark records the SLOT choice of this maximum
        hAlpha = float('-inf') 
        mark=0
        
        # initialization is necessary for board closing to full 
        nexts = next(board, side=side)
        if nexts!=[]:
            mark = nexts[0][1] 

        # traverse loop   
        for son in nexts:
            #  calculate the value by call move function of this node recursively, recort its value by v
            v, choice = move(son[0], depth=depth+1, alpha=alpha, beta=beta)
            # record its SLOT choice by choice 
            choice = son[1]

            # refresh hAlpha and mark
            if hAlpha<v:
                hAlpha=v
                mark=choice
            
            #judge if can do alpha-beta pruning by comparing current hAlpha and its original beta
            # if it meets the condition of pruning, the evaluation value of this node determains, just return
            if hAlpha>=beta and depth!=0: # beta is the minimal beta of ancestral minimum nodes
                return hAlpha, mark
            # one son node got, next son node's original alpha value should be refleshed

            if alpha<v:
                alpha=v
        # pruning doesn't happen and the traverse over, return its final alpha value
        return hAlpha, mark

    #Human side is just similar to AI side
    if side==Human_token:  
        if legal_move_left(board)==False:
            return 0, 0

        hBeta=float('inf') 
        mark=0

        nexts = next(board, side=side)
        if nexts!=[]:
            mark= nexts[0][1]
       
        for son in nexts:
            v, choice = move(son[0], depth=depth+1, alpha=alpha, beta=beta)
            choice = son[1]
 
            if hBeta>v:
                hBeta=v
                mark=choice

            if hBeta<= alpha:
                return hBeta, mark

            if beta>v:
                beta=v

        return hBeta, mark

#评估函数
  #push the current board to each side's extreme, usually use the number of defined advantage unit of winning, then:

    #calculate advantage value of AI: vAI
def evaluate(board):
    vAI = 0
    boardAI = copy.deepcopy(board)
    for i in SLOTS:
        if board[i] == Open_token:
            boardAI[i] = 1
    for triad in WINNING_TRIADS:
        triad_sum = boardAI[triad[0]] + boardAI[triad[1]] + boardAI[triad[2]]
        if triad_sum == 3:
            vAI += 1
            
    #calculate advantage value of Human: vHuman
    vHuman = 0 
    boardHuman = copy.deepcopy(board)
    for i in SLOTS:
        if board[i] == Open_token:
            boardHuman[i] = -1
    for triad in WINNING_TRIADS:
        triad_sum = boardHuman[triad[0]] + boardHuman[triad[1]] + boardHuman[triad[2]]
        if  triad_sum == -3:
            vHuman += 1    
    return vAI - vHuman


HUMAN = 1
COMPUTER = 0
def main():
    """主函数,先决定谁是X(先手方),再开始下棋"""
    next_move = HUMAN 
    opt = input("请选择先手方，输入X表示你先手，输入O表示AI先手：")
    if opt == "X":
        next_move = HUMAN
    elif opt == "O":
        next_move = COMPUTER
    else:
        print("输入有误，默认为你先手")
    
    # 初始化当前棋盘为空棋盘
    board = [Open_token for i in range(9)]

    # 开始下棋
    while legal_move_left(board) and winner(board) == Open_token: #没谁赢并且可以下
        print()
        print("当前棋局为:")
        print_board(board)
        
        if next_move == HUMAN:
            try:
                print("\n")
                humanmv = int(input("请输入你要落子的位置(0-8)："))
                if board[humanmv] != Open_token:
                    print("你输入的位置已经有子了,换个地方下吧")
                    continue
                board[humanmv] = Human_token
                next_move = COMPUTER
            except:
                print("输入有误，请重试")
                continue
            
        if next_move == COMPUTER:
            if winner(board) == Human_token:
                break
            mymv = (move(board))[1]
            print("AI最终决定下在", mymv)
            board[mymv] = AI_token
            next_move = HUMAN

    # 输出结果
    print_board(board)
    print()
    print("游戏结束 "+["平局", "AI赢了！", "你赢了！"][winner(board)])


if __name__ == '__main__':
    main()