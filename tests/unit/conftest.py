import pytest

pytest_plugins = ["pytester"]

@pytest.fixture(autouse=True)
@pytest.mark.asyncio_event_loop
def setup_asyncio_loop():
    pass
