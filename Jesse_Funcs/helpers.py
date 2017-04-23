from __future__ import division
import time
import operator
import itertools
import numpy as np
import sys


def loop(func, iters, outs, savefile=None, kwargs={}):
    '''
    A function to iterate over the dictionary 'iter_dict' and run 'function' with
    'kwargs', saving and returning 'outs' into 'savefile'
    
    Params:
    -------
    func
        (function) Takes kwargs, and iters keywords, returns outs
    iters
        (list of tuples) [('keyword of iterable': iterator), ...]
        Needs to be a list of tuples, not dict, to preserve order.
        Takes the cartesian product. (left is outer loop, right is inner)
    kwargs
        (dict) {kw:value}
    outs
        (list of strings and None) Names of variables returned by function.
        You can put None to not save/return a variable.
    savefile
        (string) Name of '.npz' file to save
        
    Returns:
    --------
    out_dict
        (dict) {*outs:array(n_iter_left,...,n_iter_right[, n_outs if same for each function call],
        iters:iters, func:func, *kwargs:*kwargs}
            Returns dictionary with arrays of size (n_iter_left,...,n_iter_right, n_return)
            
    Examples
    --------
    >>> def test(x, y, z):
    >>>     a = x + y + z
    >>>     b = x*y*z
    >>>     return a, b
    >>> 
    >>> func = lambda x, y: test(x, y, z=1)  # kwargs = {}
    >>> func = test
    >>> iters = [('x', range(3)), ('y', range(5))]
    >>> kwargs = dict(z=1)
    >>> outs = ['a', 'b']
    >>> 
    >>> res = loop(func, iters, outs, 'test.npz', kwargs)
    >>> res['b']
    >>> array([[0, 0, 0, 0, 0],
    >>>    [0, 1, 2, 3, 4],
    >>>    [0, 2, 4, 6, 8]])

    Ex. 2
    -----
    >>> def test(x, y, z):
    >>>     a = x + y + z
    >>>     b = x*y*z
    >>>     return a, b
    >>> 
    >>> func = lambda x, y: test(x, y, z=1)
    >>> iters = [('x', range(3)), ('y', range(5))]
    >>> outs = ['a', 'b']
    >>> 
    >>> res = loop(func, iters, outs)
    >>> helpers.dict2global(res)
    '''
    
    #Make sure iters is a list of tuples
    if len(iters[0]) < 2:
        iters = [iters]
        
    #Create Cartesian Product
    iter_keys, iter_values = zip(*iters)
    n_iters = [len(v) for v in iter_values]
    values = itertools.product(*iter_values)
    
    #Create ouput dictionary
    out_dict = {}
    for out in outs:
        if out:
            out_dict[out] = []    

    for value in values:
        #Combine keyword dictionaries
        kwargs_iter = dict(zip(iter_keys, value))
        kwargs_all = dict(kwargs.items() + kwargs_iter.items())
                
        #Run the Function
        result = func(**kwargs_all)
        
        #Append the results for each iteration        
        for res, out in zip(result, outs):
            if out:
                out_dict[out].append(res)        


       
    #Turn Everything Into an Array
    for k, v in out_dict.items():
        temp = np.array(v)

        # Add extra dim if all results the same size
        if temp.ndim > 1:
            new_shape = n_iters + [temp.shape[-1]]
        else:
            new_shape = n_iters

        # Reshape the array
        out_dict[k] = temp.reshape(new_shape)



    #Add iters (preserves order) and kwargs
    iters_dict = dict(iters)
    iters_dict['order'] = iter_keys
    out_dict['iters_dict'] = np.array([iters_dict])
    out_dict['kwargs_dict'] = np.array([kwargs])
    
                
    #Save!
    if savefile:
        np.savez(savefile, **out_dict)
        
    return out_dict






def add_to_database(result, index, goal, filename='test.npz'):
    ''' 
    Checks if result is better than database at index, and if so, replaces
    the database at index for each of outs and saves it to disk. Requires that all 
    dictionary values in savefile be of the same length (index is same).

    Parameters
    ----------
    result
        (dict)
    index
        (TUPLE)
    goal
        (tuple) ('variable','max' or 'min')
    outs
        (list of strings)
    filename
        (string to .npz file)

    Example
    -------
    def test(x, y, z):
    a = x + y + z
    b = x*y*z
    return a, b
    
    func = lambda x, y: test(x, y, z=1)
    iters = [('x', range(3)), ('y', range(5))]
    outs = ['a', 'b']     
    res = loop(func, iters, outs, filename='test.npz')

    r = {'a':11, 'b':12}
    goal = ('a','max')
    index = (0,0)
    outs = ['a', 'b']
    dataz = add_to_database(r, index, goal, 'test.npz')
    '''
    try:
        database = np.load(filename)
    except IOError:
        print 'Wrong filename'
        return
        
    goal_var, maxmin = goal
    goal_cmp = [operator.lt, operator.gt][['min', 'max'].index(maxmin)]
    
    data_val = database[goal_var][index]
    res_val = result[goal_var]

    # Check if better
    if goal_cmp(res_val, data_val):

        # Copy the Dict (Necessary for new save for some reason)
        temp_dict = {}
        for k, v in database.items():
            temp_dict[k] = v

        # Overwrite data
        for out in result.keys():
            if out not in ['iters_dict', 'order']:
                temp_dict[out][index] = result[out]

        # Save
        np.savez(filename, **temp_dict)





def dict2global(*dicts, **kwargs):
    '''
    Imports dictionary to global variables.
    NEED to import directly to work properly!
    ex. 
    from ... import dict2global
    
    Keywords:
    ---------
    recursive
        (bool) If True, globalize all subdictionaries.
    '''
    #A hack to allow kwords and variable args
    if not locals().has_key('recursive'):
        recursive = True

    #Create Global Injector
    Global = _global_injector()

    for dict_ in dicts:

        for k, v in dict_.items():
            if recursive:

                if type(v) == dict:
                    dict2global(v)

                else:
                    Global[k] = v



class _global_injector:
    '''Inject into the *real global namespace*, i.e. "builtins" namespace or "__builtin__" for python2.
    Assigning to variables declared global in a function, injects them only into the module's global namespace.
    >>> Global= sys.modules['__builtin__'].__dict__
    >>> #would need 
    >>> Global['aname'] = 'avalue'
    >>> #With
    >>> Global = global_injector()
    >>> #one can do
    >>> Global.bname = 'bvalue'
    >>> #reading from it is simply
    >>> bname
    bvalue

    '''
    def __init__(self):
        try:
            self.__dict__['builtin'] = sys.modules['__builtin__'].__dict__
        except KeyError:
            self.__dict__['builtin'] = sys.modules['builtins'].__dict__
    def __setattr__(self,name,value):
        self.builtin[name] = value
    def __setitem__(self,name,value):
        self.builtin[name] = value







def txt2npz(filename, kwargs=None, style='PulseData'):
    """
    Turns a .txt file into a .npz file 
    .txt file is tab seperated from Pulse data

    Param:
    ------
    filename
      (string) No .txt suffix 
    """
    print "Importing..."

    if kwargs:
        data_array = np.genfromtxt(filename+'.txt', **kwargs)
    elif style == 'PulseData':
        data_array = np.genfromtxt(filename+'.txt', skip_header=3, names=True,
                                   dtype='i8,f8,S5,f8,f8,f8,f8,f8,f8')

    temp = {}
    for key in data_array.dtype.fields.keys():
        temp[key] = data_array[key]
        print key

    np.savez(filename+'.npz', **temp)
    print 'Done!'





def find_peaks(x, eps=2e-4):
    '''Given an array x, find the indices of all the peaks where dx < eps'''
    peak = np.logical_and( np.logical_and(x > np.append(0, x[:-1]), x > np.append(x[1:],0)), x > eps)
    return np.where(peak)

#==============================================================================
# ### random utilities ###
#==============================================================================
class tictoc():
    """A class for timing operations similar to matlab's tic() and toc() funcs
    """
    def __init__(self):
        self.t = 0
    def tic(self):
        self.t = time.time()
    def toc(self):
        return time.time()-self.t