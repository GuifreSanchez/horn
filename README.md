GitHub repository for the project "On the optimization landscape of Horn's problem"
conducted under the supervision of Andreea-Alexandra Musat and 
done in collaboration with the Chair of Continuous Optimization at EPFL
led by Dr. Nicolas Boumal

    main.py: contains a number of relevant numerical experiments for this work, 
which appear mostly commented and with a brief description at the beginning. 

    problems.py: contains functions that define different problem types, which correspond
to different choices of the matrices A, B, C (in general, these will be random solvable and
non-solvable instances of the problem).

    experiments.py: contains code which is core to many of the experiments ultimately implemented
in main.py, such as running n optimizations, gathering results, plotting cost and gradient norm values
against iteration number, etc.

    tests.py: contains a battery of initial sanity-check tests to make sure the optimization problem objects
used in the rest of the repository are correctly set. 


Author: Guifré Sánchez
Contact: guifre.sancheziserra@epfl.ch

