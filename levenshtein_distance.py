#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:06:36 2023

@author: Leo
"""
import numpy as np

def wer(correct, guess): 
# This function calculates the Levenshtein distance using the Wagner-Fischer 
#   method. The Levenshtein distance measures the minimum number of insertions,
#   deletions, and substitutions to get from one string to another

    matrix = np.zeros((len(guess)+1, len(correct)+1))
    # The Wagner-Fischer algorithm calculates LD by first making a matrix with 
    #   the correct string on one axis and the guess string on the other. Each
    #   axis also has an empty character added to the beginning, hence why each
    #   length is +1 the length of the string. Here the guess string is the y-
    #   axis and correct is the x-axis

    for i in range(1, len(guess)+1):
        matrix[i,0] = i 
    for j in range(1, len(correct)+1):
        matrix[0,j] = j
    # These two loops calculate the LD to get from an empty string to one of 
    #   the input strings. This always equals the length of the nonzero string
        
    for j in range(1,len(correct)+1):
        for i in range(1,len(guess)+1):
        # The WFA works by making a table with the correct string on one axis 
        #   and guess string on the other. Then for each cell it calculates the
        #   edit distance to get from the substring of the correct string to
        #   the substring of the other string. So for example, assuming the top
        #   left box in the matrix is [0,0] and the guess string is " bat"
        #   while the correct string is " cat", then cell [1,2] measures the
        #   edit distance from " c" to " ba", and [2,1] measures the edit 
        #   distance from " ca" to " b" (note again the initial null character)
        
            if correct[j-1] == guess[i-1]:
                (isdiff := 0)
            else:
                (isdiff := 1)
            # This code treats it as a substitution when the two strings 
            #   have the same character at a certain position, but gives it a
            #   'cost' of 0. Now that I reread it I'm not sure if this is any
            #   better than saying:
            #       if correct[j-1] == guess[i-1]:
            #           matrix[i,j] = matrix[i-1,j-1]
            
            matrix[i,j] = min(matrix[i-1, j] + 1,           # Deletion
                              matrix[i, j-1] + 1,           # Insertion
                              matrix[i-1,j-1] + isdiff)     # Substitution
            # Moving down the matrix is deletion, moving right is insertion,
            #   and moving both down and left is substitution
            
    return matrix[len(guess), len(correct)] / max(len(correct), len(guess))
    # The bottom-right cell represents the LD for the complete strings. The max
    #   LD always equals the length of the longer of the two strings, so I
    #   divide the LD in order to get a percent score. 0.0 is 'identical', 1.0
    #   is 'nothing in common'. Comment out everything after 'matrix' to see
    #   the entire table, which might make it clearer what is going on


# print(wer("kitten", "sitting"))
# print(wer("Saturday", "Sunday"))

# The function seems to be working properly. The matrices it produces for 
#   "kitten"/"sitting" and "Saturday"/"Sunday" are identical to the ones on
#   Wikipedia, and it gives common-sense answers for short test strings I've 
#   put in. It's also pretty fast. As it's written right now it's pretty 
#   heavily based on the Wikipedia pseudocode, which makes a couple choices 
#   that seem weird to me, maybe to make it easier to understand. I'm going to 
#   try making it more compact and also to implement it in the code we have



