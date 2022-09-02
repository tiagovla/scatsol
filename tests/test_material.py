import pytest
from scatsol.material import Material, Medium


def test_material_object() -> None:
    diel = Material(epsilon_r=3.0, mu_r=1.0)
    assert diel.epsilon_r == 3.0
    assert diel.mu_r == 1.0


def test_material_medium() -> None:
    diel = Material(epsilon_r=1, mu_r=1)
    med = Medium(material=diel, frequency=300e6)
    print(med.eta)
    assert pytest.approx(med.eta, abs=0.5) == 376.99111843
