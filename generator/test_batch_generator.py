#!/usr/bin/python3
#
# test_batch_generator.py - Test for the batch generator
#
# Frank Blankenburg, Nov. 2017
#

import argparse
import random
import cProfile
import pstats
import sys
import time

import generator.generator
import generator.background
import generator.fixture
import generator.sheetmetal


#----------------------------------------------------------------------------------------------------------------------
# Test generator run time
#
def test_run_time (source, args):

    start_time = time.time ()

    run = 0
    for batch in generator.generator.batch_generator (source, args.batchsize):

        intermediate_time = time.time ()
        duration = int ((intermediate_time - start_time) * 1000)

        run += 1
        if run == args.runs:
            break

        print('\rRun {run} / {total}: {time} ms'.format (run=run + 1, total=args.runs, time=duration), end='\r')

    print ()

    end_time = time.time ()

    duration = int ((end_time - start_time) * 1000)

    n = args.runs * args.batchsize

    print ('Generated data sets: {n}'.format (n=n))
    print ('Execution time : {duration} ms'.format (duration=duration))
    print ('Execution time per dataset: {duration} ms'.format (duration=(duration / n)))

#----------------------------------------------------------------------------------------------------------------------
# Profile generator execution
#
def test_profile (generator, args):

    pr = cProfile.Profile ()
    pr.enable ()

    test_run_time (generator, args)

    pr.disable ()

    stats = pstats.Stats (pr, stream=sys.stdout).sort_stats ('cumulative')
    stats.print_stats ()

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
    parser.add_argument ('-p', '--profile',   action='store_true', default=False, help='Profile run')
    parser.add_argument ('-r', '--runs',      type=int, default=1, help='Number of runs for timing measurement')
    parser.add_argument ('-b', '--batchsize', type=int, default=5, help='Batchsize')

    generator.background.BackgroundGenerator.add_to_args_definition (parser)
    args = parser.parse_args ()

    parts = [generator.background.BackgroundGenerator.create (args),
             generator.sheetmetal.SheetMetalGenerator (args),
             generator.fixture.FixtureGenerator (args)]

    source = generator.generator.StackedGenerator (args, parts)
    source.set_use_threading (False)

    if args.profile:
        test_profile (source, args)
    else:
        test_run_time (source, args)
