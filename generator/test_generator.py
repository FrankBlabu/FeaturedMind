#!/usr/bin/python3
#
# test_generator.py - Test for the combined generator classes
#
# Frank Blankenburg, Dec. 2017
#

import argparse
import random

import common.utils
import generator.generator
import generator.background
import generator.fixture
import generator.sheetmetal



#--------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()

    parser.add_argument ('-x', '--width',     type=int, default=512,  help='Width of the generated images')
    parser.add_argument ('-y', '--height',    type=int, default=512,  help='Height of the generated images')
    parser.add_argument ('-t', '--threading', action='store_true', default=False, help='Use threading')

    generator.background.BackgroundGenerator.add_to_args_definition (parser)
    args = parser.parse_args ()

    parts = [generator.background.BackgroundGenerator.create (args),
             generator.sheetmetal.SheetMetalGenerator (args),
             generator.fixture.FixtureGenerator (args)]

    source = generator.generator.StackedGenerator (args, parts)
    source.set_use_threading (args.threading)

    image, masks = source.generate ()

    common.utils.show_image ([image, 'Fixture'], [common.utils.mask_channels_to_image (masks), 'Mask'])
