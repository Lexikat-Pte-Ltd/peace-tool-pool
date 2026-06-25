from peace_tool_pool.knowledge import Bounds, KnowledgeRequest
from peace_tool_pool.knowledge.providers.earthengine import (
    EarthEngineLandcoverProvider,
    EarthEnginePopulationDensityProvider,
)


class FakeInfoValue:
    def __init__(self, value):
        self.value = value

    def getInfo(self):
        return self.value


class FakeReduceResult:
    def __init__(self, values):
        self.values = values

    def get(self, key):
        return FakeInfoValue(self.values[key])


class FakeImage:
    def __init__(self, reduce_values):
        self.reduce_values = reduce_values

    def mosaic(self):
        return self

    def clip(self, region):
        return self

    def reduceRegion(self, **kwargs):
        return FakeReduceResult(self.reduce_values)


class FakeRegion:
    def area(self):
        return FakeInfoValue(2_000_000)


class FakeGeometry:
    @staticmethod
    def Rectangle(bounds):
        return FakeRegion()


class FakeReducer:
    @staticmethod
    def frequencyHistogram():
        return "frequencyHistogram"

    @staticmethod
    def sum():
        return "sum"


class FakeEarthEngine:
    Geometry = FakeGeometry
    Reducer = FakeReducer

    def __init__(self, reduce_values):
        self.reduce_values = reduce_values
        self.initialized_projects = []

    def Initialize(self, project=None):
        self.initialized_projects.append(project)

    def ImageCollection(self, dataset_id):
        return FakeImage(self.reduce_values)


def test_earthengine_landcover_provider_shapes_distribution():
    fake_ee = FakeEarthEngine({"Map": {"10.0": 3, "20.0": 1}})
    provider = EarthEngineLandcoverProvider(ee_module=fake_ee, project="test-project")

    item = provider.query(
        KnowledgeRequest(bounds=Bounds(min_lon=-1, min_lat=1, max_lon=2, max_lat=3))
    )[0]

    assert fake_ee.initialized_projects == ["test-project"]
    assert item.key == "landcover_distribution"
    assert item.value == {"Trees": 75.0, "Shrubland": 25.0}
    assert item.record_count == 2


def test_earthengine_population_provider_shapes_density():
    fake_ee = FakeEarthEngine({"population": 1000})
    provider = EarthEnginePopulationDensityProvider(ee_module=fake_ee)

    item = provider.query(
        KnowledgeRequest(bounds=Bounds(min_lon=-1, min_lat=1, max_lon=2, max_lat=3))
    )[0]

    assert item.key == "population_density"
    assert item.value == {
        "population_total": 1000,
        "area_km2": 2.0,
        "density_people_per_km2": 500.0,
        "label": "500.0 people/km^2",
    }
