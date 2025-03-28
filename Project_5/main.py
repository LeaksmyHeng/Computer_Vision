"""
Leaksmy Heng
March 22, 2025
CS5330
Project5: Recognition using Deep Networks
"""

import logging

from Project_5 import gettingData, buildingNetworks, constants


logger = logging.getLogger(__name__)


def main():
    # Plotting the training data
    data = gettingData.Project()
    data.plot()





if __name__ == "__main__":
    main()
