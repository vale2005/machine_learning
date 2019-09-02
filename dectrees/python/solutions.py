import monkdata as m
import dtree as d
import drawtree_qt5 as qt
import random
import numpy as np
import matplotlib.pyplot as plt

def assignment1(): 
    entropy1 = d.entropy(m.monk1)
    entropy2 = d.entropy(m.monk2)
    entropy3 = d.entropy(m.monk3)

    print("Entropy of first dataset is %.6f" % entropy1)
    print("Entropy of second dataset is %.6f" % entropy2)
    print("Entropy of third dataset is %.6f" % entropy3)


def assignment3():
    gains1 = [d.averageGain(m.monk1, m.attributes[i]) for i in range(0, 6)]
    print("The gains of entropy on the first dataset per attribute is %s" % gains1)
    gains2 = [d.averageGain(m.monk2, m.attributes[i]) for i in range(0, 6)]
    print("The gains of entropy on the second dataset per attribute is %s" % gains2)
    gains3 = [d.averageGain(m.monk3, m.attributes[i]) for i in range(0, 6)]
    print("The gains of entropy on the third dataset per attribute is %s" % gains3)

def assignment5_id3():
    t1=d.buildTree(m.monk1, m.attributes)
    #qt.drawTree(t1)
    print(1 - d.check(t1, m.monk1test))
    t2=d.buildTree(m.monk2, m.attributes)
    print(1 - d.check(t2, m.monk2test))
    #qt.drawTree(t2)
    t3=d.buildTree(m.monk3, m.attributes)
    print(1 - d.check(t3, m.monk3test))
    #qt.drawTree(t3)

fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def assignment7():
    mean_of_monks = []
    var_of_monks = []
    for x in range(1,4):
        (monk, test) = get_monk_by_index(x)
        curr_means, curr_vars = [], []
        for frac in fractions:
            val = gen_validate_data(monk, test, frac)
            curr_means.append(np.mean(val))
            curr_vars.append(np.var(val))
        mean_of_monks.append(curr_means)
        var_of_monks.append(curr_vars)
    drawPlots(mean_of_monks, var_of_monks)


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def get_monk_by_index(index):
    switcher = {
        1: (m.monk1, m.monk1test),
        2: (m.monk2, m.monk2test),
        3: (m.monk3, m.monk3test)
    }
    return switcher.get(index)

def gen_validate_data(monkset, monktest, fraction):
    validation_values = []
    for x in range(1, 100):
        train, valid = partition(monkset, fraction)
        tree = d.buildTree(train, m.attributes)
        pruned = get_pruned(tree, valid)
        validation_values.append(1-d.check(pruned, monktest))
    return validation_values

def get_pruned(tree, validation_set):
    number_of_prunes = 0
    best_tree = tree
    better_than_original = True
    best_check = d.check(best_tree, validation_set)
    while better_than_original:
        pruned_trees = d.allPruned(best_tree)
        better_than_original = False
        for t in pruned_trees:
            curr_check = d.check(t, validation_set)
            if curr_check > best_check:
                number_of_prunes += 1
                best_check = curr_check
                better_than_original = True
                best_tree = t
    return best_tree

def drawPlots(mean_of_monks, var_of_monks):
    x = list(range(len(mean_of_monks[0])))  
    total_width, n = 0.8, 3  
    width = total_width / n  
    plt.bar(x, var_of_monks[0], width=width, label='Monk1',fc = 'y')
    for i in range(len(x)):  
        x[i] = x[i] + width  
    plt.bar(x, var_of_monks[2], width=width, label='Monk3',tick_label = fractions,fc = 'b')
    plt.xlabel("Fraction")
    plt.ylabel("Variance of error")
    plt.title("Variance of Error over 1000 Iterations on Different Fractions ")
    plt.legend()  
    plt.show() 

def main():
  assignment1()
  print("\n")
  assignment3()
  print("\n")
  assignment5_id3()
  print("\n")
  assignment7()

if __name__== "__main__":
  main() 