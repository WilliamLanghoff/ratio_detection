"""
Copyright [2017] [William Langhoff]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Simulation code used in:
    'Chemosensation and Potential Neuronal Mechanism of Ratio Detection in a Copepod'
    'William Langhoff, Peter Hinow, J. Rudi Strickler, Jeannette Yen'

Implements a simple conductance based neural model. Optimizes synaptic
weights through simulated annealing to find a configuration which
detects a ratio of two compounds.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import brian2 as b2
from brian2 import ms, Hz

rc('text', usetex=True)


def run_sim(r1, r2, ww, sim_time=250*ms, plot=False):
    '''
    Creates the neural model and runs the simulation. Returns the number of 
    spikes from the LNgen Neuron. Use plot=True to see the behavior of the
    local neurons as a function of time. 
    '''
    
    b2.start_scope()

    # Time Constants and reversal potentials have been hardcoded.
    ln_eqs = '''
    dv/dt = ((-80 - v) + ge * (0 - v) + gi * (-80 - v))/(20*ms) : 1
    dge/dt = -ge/(5*ms) : 1
    dgi/dt = -gi/(10*ms) : 1
    '''

    LNa = b2.NeuronGroup(1, model=ln_eqs, threshold='v>=-50', reset='v=-60',
                         method='rk4')
    LNa.v = -80

    LNb = b2.NeuronGroup(1, model=ln_eqs, threshold='v>=-50', reset='v=-60',
                         method='rk4')
    LNb.v = -80

    LNgen = b2.NeuronGroup(1, model=ln_eqs, threshold='v>=-50', reset='v=-60',
                           method='rk4')
    LNgen.v = -80
    
    ORN1 = b2.PoissonGroup(100, r1*Hz)
    ORN2 = b2.PoissonGroup(100, r2*Hz)
    
    s1a = b2.Synapses(ORN1, LNa, 'w : 1 (constant)',
                      on_pre='''
                      ge_post += w
                      ''')
    s1a.connect()
    s1a.w = ww[0]
    s1a.delay=1*ms

    s1gen = b2.Synapses(ORN1, LNgen, 'w : 1 (constant)',
                        on_pre='''
                        ge_post += w
                        ''')
    s1gen.connect()
    s1gen.w = ww[1]
    s1gen.delay=1*ms

    s2b = b2.Synapses(ORN2, LNb, 'w : 1 (constant)',
                      on_pre='''
                      ge_post += w
                      ''')
    s2b.connect()
    s2b.w = ww[0]

    s2gen = b2.Synapses(ORN2, LNgen, 'w : 1 (constant)',
                        on_pre='''
                        ge_post += w
                        ''')
    s2gen.connect()
    s2gen.w = ww[1]
    s2gen.delay=1*ms

    s_ab = b2.Synapses(LNa, LNb, 'w : 1 (constant)',
                       on_pre='''
                       gi_post += w
                       ''',
                       on_post='''
                       gi_pre -= w
                       ''')
    s_ab.connect()
    s_ab.w = ww[2]

    s_a_gen = b2.Synapses(LNa, LNgen,
                          '''
                          w3 : 1 (constant)
                          w4 : 1 (constant)
                          ''',
                          on_pre='''
                          gi_post += w4
                          ''',
                          on_post='''
                          gi_pre += w3
                          ''')
    s_a_gen.connect()
    s_a_gen.w3 = ww[3]
    s_a_gen.w4 = ww[4]
    s_a_gen.delay = '2*ms'

    s_b_gen = b2.Synapses(LNb, LNgen,
                          '''
                          w3 : 1 (constant)
                          w4 : 1 (constant)
                          ''',
                          on_pre='''
                          gi_post += w4
                          ''',
                          on_post='''
                          gi_pre += w3
                          ''')
    s_b_gen.connect()
    s_b_gen.w3 = ww[3]
    s_b_gen.w4 = ww[4]
    s_a_gen.delay = '2*ms'

    SM = b2.SpikeMonitor(LNgen)
    if plot:
        Mgen = b2.StateMonitor(LNgen, 'v', record=0)
        Ma = b2.StateMonitor(LNa, 'v', record=0)
        Mb = b2.StateMonitor(LNb, 'v', record=0)
    b2.run(sim_time)

    if plot:
        plt.plot(Mgen.t/ms, Mgen[0].v, Mgen.t/ms, Ma[0].v, Mgen.t/ms, Mb[0].v)
        plt.legend(['LNgen', 'LNa', 'LNb'])

    return(len(SM.t))


def freqs(ww=None):
    '''Returns 10x10 matrix of frequencies (Hz)'''
    fm = np.zeros((10, 10))
    for i in range(10):
        r1 = 10 * (1.3**i)
        for j in range(i, 10):
            r2 = 10 * (1.3**j)
            spikes = run_sim(r1, r2, ww)
            fm[i, j] = spikes*4
            fm[j, i] = fm[i, j]
    return fm


def conv_kernel():
    '''Matrix to be convolved with results. Originally chosen heuristically in
    Zavada et al (2011), reused here for our model.'''
    a, b, c = 18, 1.25, 0.3
    tm = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            tm[i, j] = a * (np.exp((-((2*(1.3**i) - 2*(1.3**j))**2))/(b**2)) - c)
    return tm


def cost(ww):
    fm = freqs(ww)
    ck = conv_kernel()
    ret = 0
    for i in range(10):
        for j in range(10):
            ret -= fm[i, j] * ck[i, j]
    return ret


def minimize_cost(num_iters, search_dist):
    temp = 1.0
    ww = np.random.rand(5)
    best_cost = cost(ww)
    old_cost = best_cost
    best_ww = ww
    print('Best cost %i: (%.05f, %.05f, %.05f, %.05f, %.05f)' %
          (best_cost, ww[0], ww[1], ww[2], ww[3], ww[4]))

    for idx in range(num_iters):
        print('iter %i' % idx)
        dw = np.random.randn(5)
        dw /= np.linalg.norm(dw)
        dw *= search_dist

        ww_new = ww + dw
        for i in range(5):
            ww_new[i] = np.abs(ww_new[i])

        cc = cost(ww_new)
        if cc < old_cost:
            old_cost = cc
            ww = ww_new
        elif np.random.rand() < np.exp((old_cost - cc)/temp):
            old_cost = cc
            ww = ww_new

        if cc < best_cost:
            best_cost = cc
            best_ww = ww
            print('Best cost %i: (%.05f, %.05f, %.05f, %.05f, %.05f)' %
                  (best_cost, ww[0], ww[1], ww[2], ww[3], ww[4]))
        if idx % 5 == 4:
            temp *= 0.9

    return best_cost, best_ww


def conv_kernel_plot():
    ck = conv_kernel()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(ck, origin='lower', cmap=plt.cm.gray)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('index', size=12)
    ax.set_ylabel('index', size=12)
    ax.set_title(r'Convolution Kernel \textbf{T}', size=16)
    cbar = fig.colorbar(cax)
    cbar.set_label('Weight  (a.u).', rotation=270, size=12)

    plt.show()
    fig.savefig('conv_kernel.pdf')
