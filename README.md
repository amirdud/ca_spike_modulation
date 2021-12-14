# Synaptic input and ACh modulation regulate dendritic Ca<sup>2+</sup> spike duration

The code recreates the figures from "Synaptic input and ACh modulation regulate dendritic Ca<sup>2+</sup> spike duration in pyramidal neurons, directly affecting their somatic output" by Dudai, Doron, Segev and London (2021), Journal of Neuroscience

In layer 5 cortical pyramidal neurons, the principal projection neurons of the cortex, dendritic Ca<sup>2+</sup> spikes play a key role in signal processing. Employing single-cell simulations, we reproduced the complexity of the Ca<sup>2+</sup> spike in a reduced dynamical model and identified the main factors regulating its duration. We found that while the overall shape of the Ca<sup>2+</sup> spike is stable, its duration is sensitive to synaptic perturbations and cholinergic modulation. We also showed that, counterintuitively, timed excitation may shorten the spike, while inhibition may extend it. Through a dynamical systems analysis, this work sheds light on the mechanisms controlling the Ca<sup>2+</sup> spike, a fundamental signal that amplifies excitatory input, facilitates somatic action potentials, and promotes synaptic plasticity.

![Ca2+ Spike Dynamics](https://github.com/amirdud/ca_spike_modulation/blob/main/Ca_Spike_Video.gif)

To run the scripts, install [NEURON](https://neuron.yale.edu/) and the L5PC models from [ModelDB](https://senselab.med.yale.edu/ModelDB/default) (see Materials and Methods). Then, compile the mod files using nrnivmodl through the command line.
