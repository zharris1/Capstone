from main.dataCollector.sourceOne import *
from main.dataCollector.sourceTwo import *
from main.dataCollector.sourceThree import *
from main.dataCollector.sourceFour import *
from main.recommender import * 
from main.stylist import * 
from utils.utils import *

if __name__ == '__main__':
    ToyData().trainMNIST()
    ToyData().testMNIST()
    SourceOne()
    SourceTwo()
    SourceThree()
    SourceFour()
    Stylist()
    Recommender()
    