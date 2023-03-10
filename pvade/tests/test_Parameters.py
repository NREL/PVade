import pytest
from pvade.Parameters import SimParams

@pytest.fixture()
def create_params():
    params = SimParams("test_params.yaml")

    return params

@pytest.mark.unit
def test_parameters(create_params):
	params = create_params
	print(params)

	print('yo')