import json

from data_storage.chen_data import ChenData
from experiments.experiments import Experiments
from models import LogisticRegressionModel

if __name__ == '__main__':
    filepath = '../../datasets/wsdm.csv'
    chendata = ChenData(filepath)
    chendata.transform_points()

    reg_coeffs = [1, 0.1, 0.01] #[0.00001, 0.00005] #, 0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10, 15]
    eps = 1e-5
    percentage_of_marks = [100, 80, 50] #[15, 20] #, 30, 40, 60, 80, 100]
    reg_types = ['lasso'] #, 'ridge']
    cnt_trials = 1

    boot_params = []
    boot_params_description = []
    for percent in percentage_of_marks:
        boot_params.append({
            'marks_percentage': percent,
            'cnt_trials': cnt_trials
        })

        boot_params_description.append('{}%_{}trials'.format(percent, cnt_trials))

    DS_params = [
        {
            'model': LogisticRegressionModel()
        }
    ]
    DS_params_description = ['DS']

    Raykar_params = []
    Raykar_params_description = []
    RaykarDS_params = []
    RaykarDS_params_description = []

    for reg_type in reg_types:
        for reg_coeff in reg_coeffs:
            Raykar_params.append({'model': LogisticRegressionModel(reg_type=reg_type, reg_coeff=reg_coeff)})
            Raykar_params_description.append('Raykar_{}_{}'.format(reg_type, reg_coeff))

            RaykarDS_params.append({'model': LogisticRegressionModel(reg_type=reg_type, reg_coeff=reg_coeff)})
            RaykarDS_params_description.append('RaykarDS_{}_{}'.format(reg_type, reg_coeff))

    result = Experiments(chendata).run_experiments(boot_params, boot_params_description,
                                          RaykarDS_params, RaykarDS_params_description,
                                          Raykar_params, Raykar_params_description,
                                          DS_params, DS_params_description)

    with open('result.json', 'w') as file_to:
        json.dump(result, file_to)
