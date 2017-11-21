#!/usr/bin/python3
#
# test_generator_performance.py - Test of the performance of the generator classes
#
# Frank Blankenburg, Nov. 2017
#

import argparse
import random
import cProfile
import pstats
import sys
import time

import generator.background
import generator.fixture
import generator.sheetmetal


#----------------------------------------------------------------------------------------------------------------------
# Test generator run time
#
def test_run_time (generator, runs):

    start_time = time.time ()

    for i in range (runs):
        print('\rRun {run} / {total}'.format (run=(i + 1), total=runs), end='\r')
        image, mask = generator.generate ()

    print ()

    end_time = time.time ()

    duration = int ((end_time - start_time) * 1000)

    if runs == 1:
        print ('Execution time : {duration} ms'.format (duration=duration))
    else:
        print ('Execution time : {duration} ms (avg. {steptime} ms per run)'.format (duration=duration,
                                                                                     steptime=duration / runs))

#----------------------------------------------------------------------------------------------------------------------
# Profile generator execution
#
def test_profile (generator, runs):

    pr = cProfile.Profile ()
    pr.enable ()

    test_run_time (generator, runs)

    pr.disable ()

    stats = pstats.Stats (pr, stream=sys.stdout).sort_stats ('cumulative')
    stats.print_stats ()


#----------------------------------------------------------------------------------------------------------------------
# MAIN
#
if __name__ == '__main__':

    random.seed ()

    #
    # Parse command line arguments
    #
    parser = argparse.ArgumentParser ()

    generators = [ generator.background.NoisyRectBackgroundGenerator,
                   generator.background.ImageBackgroundGenerator,
                   generator.fixture.FixtureGenerator,
                   generator.sheetmetal.SheetMetalGenerator ]

    parser.add_argument ('-x', '--width',     type=int, default=512, help='Width of the generated image')
    parser.add_argument ('-y', '--height',    type=int, default=512, help='Height of the generated image')
    parser.add_argument ('-g', '--generator', action='store', choices=[generator.TYPE for generator in generators])
    parser.add_argument ('-p', '--profile',   action='store_true', default=False, help='Profile run')
    parser.add_argument ('-r', '--runs',      type=int, default=1, help='Number of runs for timing measurement')

    generator.background.BackgroundGenerator.add_to_args_definition (parser)

    args = parser.parse_args ()

    #
    # Instantiate generator class from command line arguments
    #
    generator = None
    for candidate in generators:
        if args.generator == candidate.TYPE:
            generator = candidate (args)

    if generator is None:
        raise RuntimeError ('Unknown generator type \'{typename}\''.format (typename=args.generator))

    if args.profile:
        test_profile (generator, args.runs)
    else:
        test_run_time (generator, args.runs)
