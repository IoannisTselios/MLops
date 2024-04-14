import requests
import os


def test_endpoints():
    for i in range(10):
        image_files = os.listdir('tests/test_data')
        images = [('imagefiles', open(
            f'tests/test_data/{image_file}', 'rb')) for image_file in image_files]
        response = requests.post(
            'https://dog-breed-identification-api-k3daan6qya-ew.a.run.app/predict', files=images)
        assert response.status_code == 200, "Response code is incorrect"
