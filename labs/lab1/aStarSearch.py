import numpy as np
import random
import math
class PuzzleBoardState(object):
    def __init__(self, dim=3, random_seed=2022, data=None, parent=None):
        self.dim = dim
        self.default_dst_data = np.array([[1,2,3], [4,5,6], [7,8,0]])
        if data is None:
            init_solvable = False
            init_count = 0
            while not init_solvable and init_count<500:
                init_data = self._get_random_data(random_seed=random_seed+init_count)
                init_count += 1
                init_solvable = self._if_solvable(init_data, self.default_dst_data)
                print("init",init_data)
            data = init_data
        self.data = data
        self.parent = parent
        self.piece_x, self.piece_y = self._get_piece_index()
        
    def _get_random_data(self, random_seed):
        random.seed(random_seed)
        init_data = [i for i in range(self.dim**2)]
        random.shuffle(init_data)
        init_data = np.array(init_data).reshape((self.dim, self.dim))

        return init_data

    def _get_piece_index(self):
        index = np.argsort(self.data.flatten())[0]

        return index//self.dim, index%self.dim

    def _inverse_num(self, puzzle_board_data):
        flatten_data = puzzle_board_data.flatten()
        res = 0
        for i in range(len(flatten_data)):
            if flatten_data[i] == 0:
                continue
            for j in range(i):
                if flatten_data[j] > flatten_data[i]:
                    res += 1
        
        return res

    def _if_solvable(self, src_data, dst_data):
        assert src_data.shape == dst_data.shape, "src_data and dst_data should share same shape."
        inverse_num_sum = self._inverse_num(src_data) + self._inverse_num(dst_data)

        return inverse_num_sum%2 == 0

    def is_final(self):
        flatten_data = self.data.flatten()
        if flatten_data[-1] != 0:
            return False
        for i in range(self.dim**2 - 1):
            if flatten_data[i] != (i + 1):
                return False
        return True

    def next_states(self):
        res = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            x2, y2 = self.piece_x + dx, self.piece_y + dy
            if 0 <= x2 < self.dim and 0 <= y2 < self.dim:
                new_data = self.data.copy()
                new_data[self.piece_x][self.piece_y] = new_data[x2][y2]
                new_data[x2][y2] = 0
                res.append(PuzzleBoardState(data=new_data, parent=self))

        return res

    def get_data(self):
        return self.data

    def get_data_hash(self):
        return hash(tuple(self.data.flatten()))

    def get_parent(self):
        return self.parent
    
    def h(self):#不在位的数字数
        h = 0
        flatten_data = self.data.flatten()
        for i in range(self.dim**2-1):
            if flatten_data[i] != (i + 1):
                h += 1
        return h 

def findOpt(lst):
        opt = lst[0][0]+lst[0][1]
        index = 0
        for i in range(len(lst)):
            if lst[i][0]+lst[i][1] < opt:
                opt = lst[i][0]+lst[i][1] 
                index = i
        return index
def A_Star_Search(puzzle_board_state):
    #visited集合记录以搜索节点
    visited = set()
    #list列表记录待搜索节点
    list = []
    now = 0
    list.append((puzzle_board_state.h(), 0, puzzle_board_state))
    visited.add(puzzle_board_state.get_data_hash())
    ans = []
    while(list):
        #利用findOpt函数找到f最小的列表索引
        index = findOpt(list)
        search_state = list[index][2]
        now = list[index][1]
        h = list[index][0]
        #在列表中删除该元素
        del list[index]
        #判断是否是目标节点
        if search_state.is_final():
                while search_state.get_parent() is not None:
                    #parent节点insert进第0位，使实现从头向后输出
                    ans.insert(0, search_state)
                    search_state = search_state.get_parent()
                break
        #搜索该节点，拓展未被搜索过的子节点加入列表
        new_states = []
        new_states.extend(search_state.next_states())
        for state in new_states:
            if state.get_data_hash() in visited:
                continue
            visited.add(state.get_data_hash())
            list.append((state.h(), now+1, state))
    return ans

def bfs(puzzle_board_state):
    visited = set()

    from collections import deque
    queue = deque()
    queue.append((0, puzzle_board_state))
    visited.add(puzzle_board_state.get_data_hash())

    ans = []
    while queue:
        (now, cur_state) = queue.popleft()
        if cur_state.is_final():
            while cur_state.get_parent() is not None:
                ans.append(cur_state)
                cur_state = cur_state.get_parent()
            ans.append(cur_state) #####无父节点 直达 保证这种情况正确输出解
            break

        next_states = cur_state.next_states()
        for next_state in next_states:
            if next_state.get_data_hash() in visited:
                continue
            visited.add(next_state.get_data_hash())
            queue.append((now + 1, next_state))
    return ans

if __name__ == "__main__":
    # test_data = np.array([[5,3,6], [1,8,4], [4,2,0]])
    test_board = PuzzleBoardState()
    test_board.next_states()
    res = A_Star_Search(test_board)
    for i in res:
        print(i.data)
    print(len(res))