import random

class SourceFour():
    
    """
    
    SourceFour.py: User's Preferences such as size, fit, color, inseam, etc...

    -------------------------------------------------------------------------------------------

    Functions:
    
    """
    
    
    def __init__(self):
        pass

    def generateTestUserProfiles(self):

        '''
        This function will return four user profiles we can use for testing the models. 
        Note: this function is meant to be a substitute for a front end, if we do not develop one
        '''
        userProfiles = ['userProfileOne', 'userProfileTwo', 'userProfileThree', 'userProfileFour']
        pants = ['pantOne', 'pantTwo', 'pantThree', 'pantFour']
        shirts = ['shirtOne', 'shirtTwo', 'shirtThree', 'shirtFour']
        sizes = ['Small', 'Medium', 'Large']
        colors = ['red', 'green', 'blue', 'black']

        # Please dont judge the hardcoded, I was in a hurry
        self.testUserProfiles = {
            self.userProfileOne : {
                str(random.choice(pants)): {'size': random.choice(sizes), 'color': random.choice(colors)},
                str(random.choice(shirts)): {'size': random.choice(sizes), 'color': random.choice(colors)}
            },
            self.userProfileTwo : {
                str(random.choice(pants)): {'size': random.choice(sizes), 'color': random.choice(colors)},
                str(random.choice(shirts)): {'size': random.choice(sizes), 'color': random.choice(colors)}
            },
            self.userProfileThree : {
                str(random.choice(pants)): {'size': random.choice(sizes), 'color': random.choice(colors)},
                str(random.choice(shirts)): {'size': random.choice(sizes), 'color': random.choice(colors)}
            },
            self.userProfileFour : {
                str(random.choice(pants)): {'size': random.choice(sizes), 'color': random.choice(colors)},
                str(random.choice(shirts)): {'size': random.choice(sizes), 'color': random.choice(colors)}
            }
        }

        return self.testUserProfiles
        
