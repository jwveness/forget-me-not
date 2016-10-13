Source Code for *The Forget-me Not Process* 
============================

This C++11 source code is provided as is, for illustrative purposes only. It can be used to regenerate the "Mysterious Bag of Coins" results in the publication:

``The Forget-me-not Process, Kieran Milan, Joel Veness et al, NIPS, 2016``.	

By default, the Forget-me-not implementation uses pruning. To turn off pruning, one can toggle flags in `src/fmn.hpp`.

Any questions should be directed to Kieran Milan (kmilan@google.com) or Joel Veness (aixi@google.com).


Compiling
======

For Linux,

 1. Install CMake
 2. Install Boost 1.54 or higher
 3. From the top-level directory, run:
 
``` 
mkdir build
cd build
cmake ../src
make
```

The code has been tested on Ubuntu/Gcc and Windows 7/MSVC. 

Program  Usage
==========
 
Different experimental runs can be invoked by:
 
``fmn [random seed] [number of tasks] [switch probability] [data sequence length] [repeats]``

`fmn --help` will display the program usage.
`fmn` will run with default settings.

For example, the following command regenerates the experimental results reported in the publication (albeit with less statistical signifcance, change the last parameter to 10000 to match the reported experiments):

`./fmn 666 7 0.005 5000 200`

and will return something like:

```
Using fixed data sequence seed : 666

----------------------------------------------------
             The Mysterious Bag of Coins            
----------------------------------------------------
Experiment Parameters : 
Number of coins :        7
Switch probability :     0.005
Sequence length :        5000
Repetitions :            10000

Estimated Mean Redundancy :
PTW-KT = 157.189 +- 0.770513
KT = 783.859 +- 7.79046
FMN(KT) = 148.428 +- 0.753005

oracle loss = 2503.69 +-8.37921
```
