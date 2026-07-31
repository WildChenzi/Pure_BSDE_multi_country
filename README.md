# Pure_BSDE_multi_country
This code solves BSDEs via analogous policy improvement and fixed-point iteration, with reference to https://github.com/deeProbabilism/Multi-Country-Macro-Finance, and targets researchers interested in BSDEs and Mean Field Games.

The core of this code lies in solving BSDEs in the strict sense, i.e., those with coupled Y and Z components. Iterate over all irrelevant variables via fixed-point iterations, though convergence is not guaranteed. 

The toy model is a simple continuous-time value function iteration problem. The process is: 1.start with an initial guess of the policy function, 2.solve the corresponding BSDE to obtain the value function V given this policy, 3.iterate the policy function using V until convergence.

The three-country model references the paper at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4122454.
The core feature is that interest rate is not a solution of BSDEs. The process is: 1.initialize an interest rate guess, 2.train BSDE-associated variables q and sig_q conditional on rate, 3.update and iterate the interest rate implied by q until the rate converges. The code records interest rate convergence trajectories as well as the solved U-shaped and inverted-U-shaped sig_q profiles.

