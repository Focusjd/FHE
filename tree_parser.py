import re
import random
from tqdm import tqdm

def label_str_parser(string) -> dict:
    result = {}

    x_match = re.search(r'x\[(\d+)\] <= ([\d.]+)', string)
    if x_match:
        result["x_index"] = int(x_match.group(1))
        result["x_value"] = float(x_match.group(2))
    else:
        result["x_index"] = -1
        result["x_value"] = None

    gini_match = re.search(r'(n?gini) = ([\d.]+)', string)
    if gini_match:
        result[gini_match.group(1)] = float(gini_match.group(2))

    result["nsamples"] = int(re.search(r'nsamples = (\d+)', string).group(1))
    
    result["nvalue"] = tuple(map(int, re.search(r'value = \[(\d+), (\d+)\]', string).groups()))

    class_match = re.search(r'class = ([\d.]+)', string)
    if class_match:
        result["class"] = float(class_match.group(1))
    else:
        result["class"] = None

    return result



class TreeNode:
    def __init__(self, feature_num, threshold=None, parent=None, left=None, right=None, bi_code=0, index=-1, classes=None):
        self.feature_num = feature_num
        self.threshold = threshold
        self.left = left
        self.right = right
        self.parent = parent
        self.index = index
        self.bi_code = bi_code
        self.depth = -1
        self.classes=classes
    

    def __repr__(self):
            return f"TreeNode(feature_num={self.feature_num}, bi_code={self.bi_code})"

    def set_children(self, node):
        if self.left == None:
            self.left = node
        else:
            self.right = node

    

def find_node(index, nodes_list) -> TreeNode:
    for node in nodes_list:
        if node.index == index:
            return node
    raise Exception("Node not found: ", index)  
    



def dot_parser(dot_str, nodes_list):
    nodes = re.findall(r'(\d+) \[label="(.*?)"\]', dot_str)
    for node in nodes:
        index, label_str = node
        label_dict = label_str_parser(label_str)

        treeNode = TreeNode(
            feature_num=label_dict['x_index'],
            threshold=label_dict['x_value'],
            index=int(index),
            classes=int(label_dict['class'])
            )
        nodes_list.append(treeNode)
    
    relationships = re.findall(r'(\d+) -> (\d+)', dot_str)
    for rel in relationships:
        # x -> y
        x, y = rel
        x_node = find_node(int(x), nodes_list)
        y_node = find_node(int(y), nodes_list)
        x_node.set_children(y_node)
        y_node.parent=x_node


def build_tree(node:TreeNode, nodes_list, depth=0, bi_code=0b000001):
    node.bi_code=bi_code
    node.depth=depth
    if depth == tree_depth:
        return
    
    if node.left is None:
        node.left = TreeNode(feature_num=node.feature_num, parent=node, classes=node.classes)
        node.right = TreeNode(feature_num=node.feature_num,parent=node, classes=node.classes)
        nodes_list.append(node.left)
        nodes_list.append(node.right)

    
    depth += 1
    build_tree(node.left, nodes_list, depth, (bi_code<<1)+1)
    build_tree(node.right, nodes_list, depth, bi_code<<1)


### How to handel 0 threshold ？
def export_txt(sorted_nodes, outfile, tree_depth, model_out=None):
    max_length = max([node.bi_code for node in sorted_nodes]).bit_length()
    with open(outfile, 'w+') as f:
        for node in sorted_nodes:
            # set 0 threshold temp to dummy node
            if node.threshold is not None and int(node.threshold)<1:
                node.feature_num = -1

            bi_str = str(bin(node.bi_code)[2:]).zfill(max_length)
            fe_str = str(node.feature_num)
            if node.depth >= tree_depth:
                fe_str = str(node.classes)

            if node.threshold is not None and int(node.threshold)>0:
                th_str = ', '.join([str(i+1) for i in range(int(node.threshold))])
                output_str = ', '.join([bi_str, fe_str, th_str])
            else:
                output_str = ', '.join([bi_str, fe_str])
            f.write(output_str+ '\n')
            # print(output_str)
    
    if model_out is not None:
        with open(model_out, 'w+') as mf:
            for node in sorted_nodes:
                ot_str = ''
                if node.threshold is not None and int(node.threshold)>0:
                    ot_str = create_model_sequence(1, int(node.threshold))
                    print(ot_str)
                elif node.depth >= tree_depth:
                    random_ints = sorted(random.sample(range(1, 10), 2))
                    ot_str = create_model_sequence(random_ints[0], random_ints[1])
                else: 
                    ot_str = create_model_sequence(1, random.randint(1, 9))

                mf.write(ot_str + '\n')            


def create_model_sequence(start, end, length=8):
    sequence = []
    while len(sequence) < length:
        for i in range(start, end + 1):
            if len(sequence) < length:
                sequence.append(i)
            else:
                break
    return ' '.join(map(str, sequence))


def main_parser(in_file, out_file, tree_depth, model_clear_out=None):
    dot_string = ''
    with open(in_file, 'r') as dot_file:
        dot_string = dot_file.read()

    
    full_node_num = 2^(tree_depth+1)-1
    nodes_list = []

    dot_parser(dot_string, nodes_list)
    build_tree(nodes_list[0], nodes_list)

    sorted_nodes = sorted(nodes_list, key=lambda node: node.bi_code)
    export_txt(sorted_nodes, out_file, tree_depth, model_out=model_clear_out)

    # for item in sorted_nodes:
    #     print(item.depth)
    #     print(item.index, item.feature_num, item.threshold, item.parent, item.left, item.right, item.bi_code)

    # print(len(sorted_nodes))

if __name__ == '__main__':
    in_file = './tox21_data/ranged_data/training_tree_tox21_ranged.dot'
    out_file = './tox21_data/ranged_data/tox21_ranged_out.txt'
    tree_depth = 5
    clear_txt = './tox21_data/ranged_data/tox21_ranged_model_clear.txt'

    main_parser(in_file, out_file, tree_depth, model_clear_out=clear_txt)

