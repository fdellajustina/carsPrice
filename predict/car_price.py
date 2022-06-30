import pickle
import numpy as np
import pandas as pd

class PredictCarPrice:

    def __init__(self, car):
        self.car = car

    def prepare(self):

        label_encoder = {
            'Make': {'bmw': 0, 'gmc': 1, 'lexus': 2, 'honda': 3, 'hyundai': 4, 'audi': 5, 'toyota': 6, 'subaru': 7, 'porsche': 8,
                    'mini': 9, 'volkswagen': 10, 'isuzu': 11, 'oldsmobile': 12, 'mercedes-benz': 13, 'ford': 14, 'mitsubishi': 15,
                    'land rover': 16, 'lincoln': 17, 'cadillac': 18, 'saab': 19, 'scion': 20, 'jaguar': 21, 'hummer': 22, 
                    'suzuki': 23, 'acura': 24, 'saturn': 25, 'nissan': 26, 'dodge': 27, 'mazda': 28, 'jeep': 29, 'pontiac': 30,
                    'kia': 31, 'mercury': 32, 'chevrolet': 33, 'buick': 34, 'volvo': 35, 'infiniti': 36, 'chrysler': 37},
            'Type': {'truck': 0, 'sports': 1, 'sedan': 2, 'hybrid': 3, 'suv': 4, 'wagon': 5},
            'Origin': {'asia': 0, 'usa': 1, 'europe': 2},
            'DriveTrain': {'front': 0, 'rear': 1, 'all': 2}
        }

        new_car = pd.DataFrame([np.zeros(12)], columns=['Make', 'Type', 'Origin', 'DriveTrain', 'EngineSize', 'Cylinders',
                        'Horsepower', 'MPG_City', 'MPG_Highway', 'Weight', 'Wheelbase', 'Length'])

        new_car['EngineSize'] = self.car[0]
        new_car['Cylinders'] = self.car[1]
        new_car['Horsepower'] = self.car[2]
        new_car['MPG_City'] = self.car[3]
        new_car['MPG_Highway'] = self.car[4]
        new_car['Weight'] = self.car[5]
        new_car['Wheelbase'] = self.car[6]
        new_car['Length'] = self.car[7]
        new_car['Make'] = label_encoder['Make'][self.car[8]]
        new_car['Type'] = label_encoder['Type'][self.car[9]]
        new_car['DriveTrain'] = label_encoder['DriveTrain'][self.car[10]]
        new_car['Origin'] = label_encoder['Origin'][self.car[11]]

        return new_car.values

    def predict(self, car):
        model = pickle.load(open('model/model.pkl','rb'))
        predicted_car_value = model.predict(car)
        value = np.exp(predicted_car_value)
        return value


