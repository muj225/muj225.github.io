import re
def NEAR_regex(list_of_words,max_words_between=5,partial=False,
               cases_matter=False,greedy=True):
    '''
    Parameters
    ----------
    list_of_words : list
        A list of "words", each element is a string
        
        This program will return a regex that will look for times where word1 
        is near word2, or word2 is near word 1.
        
        It works with multiple words: You can see if words1 is near word2 or
        word3. 
        
    max_words_between : int, optional
        How many "words" are allowed between words in list_of_words. The default
        is 5, but you should consider this carefully.
        
        "words" in between are chunks of characters. "DON don don- don12 2454" 
        is 5 words.
        
        This will not allow matches if the words are separated by a newline 
        ("\n") character.
        
    partial : Boolean, optional
        If true, will accept longer words than you give. For example, if one 
        word in your list is "how", it will match to "howdy". Be careful in 
        choosing this based on your problem. Partial makes more sense with 
        longer words. 
        The default is True.
        
    cases_matter: Boolean, optional bt IMPORTANT
        If True, will return a regex string that will only catch cases where  
        words in the string have the same case as given as input to this 
        function. For example, if one word here is "Hi", then the regex 
        produced by this function will not catch "hi".
        
        If false, will return a regex string that will only work if all letters
        in search string are lowercase.
        
        The default is True.
    
    greedy: Boolean, optional
    
        If True, the regex will be "greedy" and each match is the longest string that
        satisfies the conditions (max_words_between). If False, it will find the 
        shortest string that satisfies the conditions, and continue counting.
        
        The default is True. 
     
    Warning
    -------
    See the last example. The regex is minimally greedy. 
        
         
    Unsure about speed
    -------
    I don't think this is a very "fast" function, but it should be robust. 
  
    
    Suggested use
    -------
    # clean your starting string 
    a_string_you_have = 'jack and jill went up the hill'
    
    # 1. define words and set up the regex
    words = ['jack','hill']                         
    rgx = NEAR_regex(words)                       
    
    # 2a. count the number of times the word groups are in the text near each other
    count = len(re.findall(rgx,a_string_you_have))              
    print(count)  
    
    # 2b. print the actual text matches <-- great for checking!
    text_of_matches = [m.group(0) for m in re.finditer(rgx,a_string_you_have)]
    print(text_of_matches)

    
    Returns
    -------
    A string which is a regex that can be used to look for cases where all the 
    input words are near each other.

    '''

    if len(list_of_words) > 4:
        raise ValueError('Inputs should be a short list of words (4 or less) or a short list of topics where each topic is provided using the "(word1|word2|...)" syntax.')
    
    from itertools import permutations
    
    start = r'(?:\b' # the r means "raw" as in the backslash is just a backslash, not an escape character
    
    if greedy:
        lazy = ""
    else:
        lazy = "?"
    
    if partial:
        gap   = r'[A-Za-z]*\b(?: +[^ \n\r]*){0,' +str(max_words_between)+r'}'+lazy +r' *\b'
        end   = r'[A-Za-z]*\b)'
    else:
        gap   = r'\b(?: +[^ \n]*){0,' +str(max_words_between)+r'}'+lazy +r' *\b'
        end   = r'\b)'
        
    regex_list = []
    
    for permu in list(permutations(list_of_words)):
        # catch this permutation: start + word + gap (+ word + gap)... + end
        if cases_matter: # case sensitive - what cases the user gives are given back
              regex_list.append(start+gap.join(permu)+end)           
        else: # the resulting search will only work if all words are lowercase
            lowerpermu = [w.lower() for w in permu]
            regex_list.append(start+gap.join(lowerpermu)+end)
    
    return '|'.join(regex_list)