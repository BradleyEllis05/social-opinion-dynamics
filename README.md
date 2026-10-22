# Social Opinion Dynamics Simulation

A Python simulation for my end of first year Computer Programming and Algorithms module as a part of my Engineering Mathematics degree at the University of Bristol.

## Overview

This project implements how opinions evolve within a population of interacting people, modelled as agents, on a 2D grid.

The simulation models three main behaviours:

- **Consensus:** Agents adjust their opinions through interactions with neighbouring agents.
- **Segregation:** Agents may change their location depending on the similarity of opinions in their neighbourhood, leaving a dissimilar neighbourhood.
- **Clustering:** A flood-fill algorithm identifies groups of agents with similar opinions.

## Features

- Configurable grid size and simulation duration
- Adjustable thresholds for consensus, segregation and clustering
- Random grid generation
- Visualisation using Matplotlib
- Loading and saving grid data

## Libraries

- Python
- NumPy
- Matplotlib

## Running the Project

Install the required libraries:

```bash
pip install numpy matplotlib
```

Run the simulation:

```bash
python CPA_Year1_FinalProject.py
```

## Project Background

This was an individual university programming assignment focused on implementing algorithms, working with multidimensional arrays and visualising simulation results.

## Future Improvements

Potential improvements include additional automated testing, improved performance for larger grids and more detailed analysis of how parameter choices affect the simulation outcomes.
