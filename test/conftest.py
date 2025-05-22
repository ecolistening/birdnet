def pytest_addoption(parser):
    parser.addoption(
        "--audio-dir",
        action="store",
        default="default_value",
        help="/path/to/test/dataset"
    )
