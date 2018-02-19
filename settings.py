import json
import os

DEFAULT_SETTINGS_LOCATION = './default_settings.json'

class Configuration(object):

    weight_dtype = 'f8'

    window_title = 'Hopfield Simulation'

    window_size = 800, 400

    max_size = 150

configuration = Configuration()

class Settings(Configuration):

    defaults = dict(
        size=(75, 75),
        match_percent=99.9,
        image_lib_dir='./images',
        flipped_noise=False,
        async_speed=200,
        synchronous=False,
        temperature=None,
        stop_percentage=None
    )

    def __init__(self, filename=None):
        self.filename = filename
        values = Settings.defaults.copy()
        self.values = values
        print self.values
        if filename is not None:
            if os.path.exists(filename):
                self.load(filename)

    def __getattr__(self, item):
        if item == 'values':
            return {}
        return self.values[item]

    def __setattr__(self, name, value):
        if name in Settings.defaults:
            self.values[name] = value
            return
        return super(Settings, self).__setattr__(name, value)

    def load(self, filename):
        self.filename = filename
        self.values.update(json.load(open(filename)))

    def save(self, filename):
        self.filename = filename
        json.dump(self.values, open(filename, 'w'),
                  indent=2)


settings = Settings(DEFAULT_SETTINGS_LOCATION)