"""Small command line; scientific settings live in an inspectable JSON file."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import FitConfig, Scene, SystemConfig
from .execution import exclusive_study, interrupt_on_termination, requires_large_run
from .fourier import FourierSettings
from .run import Experiment, settings_for


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('compare', 'regions', 'apertures', 'angles'))
    parser.add_argument('--profile', choices=('reference', 'retrieval', 'quick', 'test'))
    parser.add_argument('--config', type=Path, help='experiment JSON: system, scene, fit, fourier, sweep')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-large-run', action='store_true')
    args = parser.parse_args(argv)
    data = json.loads(args.config.read_text()) if args.config else {}
    unknown = data.keys() - {'profile', 'system', 'scene', 'fit', 'fourier', 'sweep', 'calibration_amplitude'}
    if unknown:
        parser.error(f'unknown configuration sections: {sorted(unknown)}')
    profile = args.profile or data.get('profile')
    if profile is None:
        parser.error('choose an explicit --profile or a profile in --config')
    system = replace(SystemConfig.profile(profile), **data.get('system', {}))
    if requires_large_run(system) and not args.allow_large_run:
        parser.error('large grid/source count requires --allow-large-run; use --profile test locally')
    scene = Scene(**data.get('scene', {}))
    fit = FitConfig(**data.get('fit', {}))
    fourier = replace(settings_for(system), **data.get('fourier', {}))
    options = data.get('sweep', {})
    with exclusive_study(), interrupt_on_termination():
        experiment = Experiment(args.output, system, scene, fit, fourier,
                                calibration_amplitude=data.get('calibration_amplitude', .02))
        experiment.manifest.update(command=args.command, profile=profile, sweep=options)
        try:
            if args.command != 'apertures':
                experiment.initialize()
            getattr(experiment, args.command)(options)
        except (Exception, KeyboardInterrupt) as error:
            experiment.finish('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', str(error))
            raise
        else:
            experiment.finish()


if __name__ == '__main__':
    main()
