#!/usr/bin/env python
from itertools import *
from collections import *
import random


def powerset(iterable):
    "From itertools: powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_candidate_worlds(num_vars):
    return powerset(set(range(num_vars)))


def get_satisfying_worlds_for_tree(tree, candidate_worlds):
    if isinstance(tree, tuple):
        if tree[0] == 'not':
            child = get_satisfying_worlds_for_tree(tree[1], candidate_worlds)
            return candidate_worlds.difference(child)
        else:
            left = get_satisfying_worlds_for_tree(tree[0], candidate_worlds)
            right = get_satisfying_worlds_for_tree(tree[2], candidate_worlds)
            if tree[1] == "and":
                return left.intersection(right)
            elif tree[1] == "or":
                return left.union(right)
            else:
                print 'syntax error', tree
    else:
        result = []
        for world in candidate_worlds:
            if tree in world:
                result.append(world)
        return set(result)


def compute_relation(left, right, universe):
    ne_intersection = left.intersection(right)
    ne_just_left = left.difference(right)
    ne_just_right = right.difference(left)
    ne_outside = universe.difference(left.union(right))
    if ne_intersection and not ne_just_right and not ne_just_left and ne_outside:
        return "="
    elif ne_intersection and ne_just_right and not ne_just_left and ne_outside:
        return "<"
    elif ne_intersection and not ne_just_right and ne_just_left and ne_outside:
        return ">"
    elif not ne_intersection and ne_just_right and ne_just_left and not ne_outside:
        return "^"
    elif not ne_intersection and ne_just_right and ne_just_left and ne_outside:
        return "|"
    elif ne_intersection and ne_just_right and ne_just_left and not ne_outside:
        return "v"
    else:
        return "#"


def create_sub_statement(universe, maxlen):
    operator = random.choice(operators)
    temp = ()
    if operator == '0' or maxlen < 2:
        temp = random.choice(list(universe))
    else:
        lhs = create_sub_statement(universe, maxlen / 2)
        rhs = create_sub_statement(universe, maxlen / 2)
        temp = tuple([lhs, operator, rhs])

    neg_or_none = random.choice(neg_or_nones)
    if neg_or_none == '0':
        return temp
    else:
        return tuple([neg_or_none, temp])


def uniq(seq, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x):
            return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


def to_string(expr, individuals):
    if isinstance(expr, int):
        return individuals[expr]
    if isinstance(expr, str):
        return expr
    elif len(expr) == 3:
        return "( " + to_string(expr[0], individuals) + " ( " + to_string(expr[1], individuals) + " " + to_string(expr[2], individuals) + " ) )"
    else:
        return "( " + to_string(expr[0], individuals) + " " + to_string(expr[1], individuals) + " )"


def get_len(tree):
    if isinstance(tree, tuple):
        accum = 0
        for entry in tree:
            accum += get_len(entry)
        return accum
    elif tree == 'and' or tree == 'or' or tree == 'not':
        return 1
    else:
        return 0

individuals = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

worlds = set(get_candidate_worlds(6))
universe = set(range(6))

neg_or_nones = ['not', '0', '0']
operators = ['and', 'or', 'and', 'or', '0', '0', '0', '0', '0']


stats = Counter()
total = 0
outputs = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [],
           6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: []}
while total < 500000:
    subuniverse = random.sample(universe, 4)
    lhs = create_sub_statement(subuniverse, 12)
    rhs = create_sub_statement(subuniverse, 12)
    sat1 = get_satisfying_worlds_for_tree(lhs, worlds)
    sat2 = get_satisfying_worlds_for_tree(rhs, worlds)
    if sat1 == worlds or len(sat1) == 0:
        continue
    if sat2 == worlds or len(sat2) == 0:
        continue
    rel = compute_relation(sat1, sat2, worlds)

    if rel != "?":
        stats[rel] += 1
        total += 1
        max_len = min(max(get_len(rhs), get_len(lhs)), 12)
        outputs[max_len].append("" + rel + "\t" + to_string(
            lhs, individuals) + "\t" + to_string(rhs, individuals))

TRAIN_PORTION = 0.85

for length in outputs.keys():
    outputs[length] = uniq(outputs[length])

    filename = 'train' + str(length)
    f = open(filename, 'w')
    for i in range(int(TRAIN_PORTION * len(outputs[length]))):
        output = outputs[length][i]
        f.write(output + "\n")
    f.close()

    filename = 'test' + str(length)
    f = open(filename, 'w')
    for i in range(int(TRAIN_PORTION * len(outputs[length])), len(outputs[length])):
        output = outputs[length][i]
        f.write(output + "\n")
    f.close()

print stats
