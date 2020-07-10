# -*- coding: utf-8 -*-
from random import random
from random import sample
from random import uniform
import numpy as np

#Sacado de: https://github.com/nathanrooy/differential-evolution-optimization
#--- FUNCTIONS ----------------------------------------------------------------+
def sphere(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total

def styblinski(x):
    a = 0
    for i in range(len(x)):
        a+=x[i]**4 - 16*x[i]**2 + 5*x[i]

    return a/2    

def ackely(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    
    sum1 = x[0]**2 + x[1]**2 
    sum2 = np.cos(c*x[0]) + np.cos(c*x[1])
    
    term1 = - a * np.exp(-b * ((1/2.) * sum1**(0.5)))
    term2 = - np.exp((1/2.)*sum2)

    return term1 + term2 + a + np.exp(1)

def beale(x):
    a=1.5
    b=2.25
    c=2.625

    term1 = a - x[0] + x[0]*x[1]
    term2 = b - x[0] + (x[0]*x[1])**2
    term3 = c - x[0] + (x[0]*x[1])**3

    return term1**2 + term2**2 + term3**2

def goldstein(x):
    
    a = x[0]+x[1]+1
    b = 19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2
    c = 2*x[0]-3*x[1]
    d = 18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2

    term1 = 1 + a**2 * b
    term2 = 30 + c**2 * d

    return term1 * term2   

def levi(x):
    a = np.sin(3*np.pi*x[0])
    b = x[0]-1
    c = 1 + np.sin(3*np.pi*x[1])**2
    d = x[1]-1
    e = 1 + np.sin(2*np.pi*x[1])**2

    return a + b**2 * c + d**2 * e

def eggholder(x):
    a = np.abs((x[0]/2)+(x[1]+47))
    b = np.abs(x[0]+(x[1]+47))

    term1 = -x[1]-47 * np.sin(np.sqrt(a))
    term2 = x[0] * np.sin(np.sqrt(b))

    return term1 - term2  

def easom(x):
    a = np.cos(x[0])*np.cos(x[1])
    b = x[0] - np.pi
    c = x[1] - np.pi

    return -a * np.exp(-(b**2 + c**2)) 

def rastrigin(x):
    a = 10
    b = 0
    for i in range(len(x)):
        b+=x[i]**2 -a*np.cos(2*np.pi*x[i])
    
    return a*len(x) + b

def shafer2(x):
    a = x[0]**2 - x[1]**2
    b = x[0]**2 + x[1]**2

    term1 = np.sin(a)**2 - 0.5
    term2 = 1 + 0.001* b

    return 0.5 + term1 / term2**2

def shafer4(x):
    a = np.abs(x[0]**2 - x[1]**2)
    b = x[0]**2 + x[1]**2

    term1 = np.cos(np.sin(a))**2 - 0.5
    term2 = 1 + 0.001* b

    return 0.5 + term1 / term2**2

def crosstrain(x):
    a = np.sin(x[0])*np.sin(x[1])
    b = np.sqrt(x[0]**2 + x[1]**2)
    c = np.abs(100 - (b/np.pi))
    
    return -0.0001*(np.abs(a*np.exp(c)))**0.1


def ensure_bounds(vec, bounds):

    vec_new = []
    # cycle through each variable in vector 
    for i in range(len(vec)):

        # variable exceedes the minimum boundary
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])

        # variable exceedes the maximum boundary
        if vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])

        # the variable is fine
        if bounds[i][0] <= vec[i] <= bounds[i][1]:
            vec_new.append(vec[i])
    #print("New vect",vec_new)        
    return vec_new


#--- MAIN ---------------------------------------------------------------------+

def minimize(cost_func, bounds, popsize, mutate, recombination, maxiter):

    #--- INITIALIZE A POPULATION (step #1) ----------------+
    
    population = []
    for i in range(0,popsize):
        indv = []
        for j in range(len(bounds)):
            indv.append(uniform(bounds[j][0],bounds[j][1]))
        population.append(indv)
            
    #--- SOLVE --------------------------------------------+

    # cycle through each generation (step #2)
    for i in range(1,maxiter+1):
        print ("GENERATION:",i)
        #print(population)
        gen_scores = [] # score keeping

        # cycle through each individual in the population
        for j in range(0, popsize):

            #--- MUTATION (step #3.A) ---------------------+
            
            # select three random vector index positions [0, popsize), not including current vector (j)
            candidates = list(range(0,popsize))
            candidates.remove(j)
            random_index = sample(candidates, 3)

            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]     # target individual

            # subtract x3 from x2, and create a new vector (x_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

            # multiply x_diff by the mutation factor (F) and add to x_1
            v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            v_donor = ensure_bounds(v_donor, bounds)

            #--- RECOMBINATION (step #3.B) ----------------+

            v_trial = []
            for k in range(len(x_t)):
                crossover = random()
                if crossover <= recombination:
                    v_trial.append(v_donor[k])

                else:
                    v_trial.append(x_t[k])
                    
            #--- GREEDY SELECTION (step #3.C) -------------+

            score_trial  = cost_func(v_trial)
            score_target = cost_func(x_t)

            if score_trial < score_target:
                population[j] = v_trial
                gen_scores.append(score_trial)
                #print( '   >',score_trial, v_trial)

            else:
                #print( '   >',score_target, x_t)
                gen_scores.append(score_target)

        #--- SCORE KEEPING --------------------------------+

        gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
        gen_best = min(gen_scores)                                  # fitness of best individual
        gen_sol = population[gen_scores.index(min(gen_scores))]     # solution of best individual

        #print ('      > GENERATION AVERAGE:',gen_avg)
        #print ('      > GENERATION BEST:',gen_best)
        print ('         > BEST SOLUTION:',gen_sol,'\n')
        
    pass

#--- CONSTANTS ----------------------------------------------------------------+

#bounds = [(-1,1),(-1,1)]            # bounds [(x1_min, x1_max), (x2_min, x2_max),...]
#popsize = 10                        # population size, must be >= 4
#mutate = 0.5                        # mutation factor [0,2]
#recombination = 0.7                 # recombination rate [0,1]
#maxiter = 20                        # max number of generations

# minimize(cost_func, bounds, popsize, mutate, recombination, maxiter)  
#minimize(sphere, [(-5,5),(-5,5)], 100, 0.1, 0.5, 100)
#minimize(ackely, [(-5,5),(-5,5)], 100, 0.1, 0.5, 100)
#minimize(beale, [(-4.5,4.5),(-4.5,4.5)], 100, 0.7, 0.5, 100)
#minimize(goldstein, [(-2,2),(-2,2)], 100, 0.1, 0.5, 100)
#minimize(levi, [(-10,10),(-10,10)], 100, 0.1, 0.5, 100)
#minimize(eggholder, [(-512,512),(-512,512)], 100, 0.3, 0.4, 100)
#minimize(styblinski, [(-5,5),(-5,5)], 100, 0.1, 0.5, 100)
#minimize(easom, [(-100,100),(-100,100)], 100, 0.1, 0.5, 100)
#minimize(rastrigin, [(-5.12,5.12),(-5.12,5.12)], 100, 0.1, 0.5, 100)
#minimize(shafer2, [(-100,100),(-100,100)], 100, 0.1, 0.5, 100)
#minimize(shafer4, [(-100,100),(-100,100)], 100, 0.4, 0.3, 200)
minimize(crosstrain, [(-10,10),(-10,10)], 100, 0.1, 0.5, 100)