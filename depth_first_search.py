import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Node:

    counter = 0

    def __init__(self, name):
        assert isinstance(name, str)
        self.name = name
        Node.counter += 1

    def __str__(self):
        return self.name

    def count(self):
        return self.counter


class Edge:

    def __init__(self, src, dest):
        assert isinstance(src, Node)
        assert isinstance(dest, Node)
        self.src = src
        self.dest = dest

    def __str__(self):
        return self.src.name + '->' + self.dest.name


class Diagraph:

    def __init__(self):
        self.edges = {}

    def add_node(self, node):
        assert isinstance(node, Node)
        if node in self.edges:
            raise ValueError('Duplicated Node')
        else:
            self.edges[node] = []

    def add_edge(self, edge):
        assert isinstance(edge, Edge)
        src = edge.src
        dest = edge.dest
        if not(src in self.edges and dest in self.edges):
            raise ValueError('Node not in graph')
        self.edges[src].append(dest)

    def children_of(self, node):
        assert isinstance(node, Node)
        return self.edges[node]

    def has_node(self, node):
        assert isinstance(node, Node)
        return node in self.edges

    def get_node(self, name):
        for n in self.edges:
            if n.name == name:
                return n
        raise NameError(name)

    def __str__(self):
        result = ''
        for src in self.edges:
            for dest in self.edges[src]:
                result += '%s -> %s \n' % (src.name, dest.name)
        return result[:-1]


class Graph(Diagraph):

    def add_edge(self, edge):
        assert isinstance(edge, Edge)
        Diagraph.add_edge(self, edge)
        reversed_ = Edge(edge.dest, edge.src)
        Diagraph.add_edge(self, reversed_)


def build_city_graph(graph_type):
    g = graph_type()
    for name in ('Boston', 'Providence', 'New York', 'Chicago', 'Denver', 'Phoenix', 'Los Angeles'):
        g.add_node(Node(name))
    g.add_edge(Edge(g.get_node('Boston'), g.get_node('Providence')))
    g.add_edge(Edge(g.get_node('Boston'), g.get_node('New York')))
    g.add_edge(Edge(g.get_node('Providence'), g.get_node('Boston')))
    g.add_edge(Edge(g.get_node('Providence'), g.get_node('New York')))
    g.add_edge(Edge(g.get_node('New York'), g.get_node('Chicago')))
    g.add_edge(Edge(g.get_node('Chicago'), g.get_node('Denver')))
    g.add_edge(Edge(g.get_node('Denver'), g.get_node('Phoenix')))
    g.add_edge(Edge(g.get_node('Denver'), g.get_node('New York')))
    g.add_edge(Edge(g.get_node('Los Angeles'), g.get_node('Boston')))
    return g


airlines = build_city_graph(Diagraph)


def print_path(path):
    """
    :param path: list of nodes
    :return: string of all visited nodes
    """
    result = ''
    for i in range(len(path)):
        result = result + str(path[i]) + ' -> '
    return result[:-4]


def depth_first_search(
        graph: Diagraph,
        start: Node,
        end: Node,
        path: list,
        shortest,
        to_print=False
):
    path = path + [start]
    if to_print:
        print('current DFS path: ', print_path(path))
    if start == end:
        return path
    for node in graph.children_of(start):
        if node not in path:
            if shortest is None or len(path) < len(shortest):
                new_path = depth_first_search(graph, node, end, path, shortest, to_print)
                if new_path is not Node:
                    shortest = new_path
        elif to_print:
            print_path('%s already visited' % node)
    return shortest


def shortest_path(graph, start, end, toPrint):
    return depth_first_search(graph, start, end, [], None, toPrint)


def test_sp(source, destination):
    sp = shortest_path(airlines, airlines.get_node(source), airlines.get_node(destination), toPrint=True)
    print()
    if sp is not None:
        print('Shortest path from', source, 'to', destination, 'is', print_path(sp))
    else:
        print('There is no path from', source, 'to', destination)


test_sp('Boston', 'Denver')




































































































































































































































































