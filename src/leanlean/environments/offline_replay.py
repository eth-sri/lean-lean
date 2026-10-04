"""Offline Lake dependency routing for the pinned Palomar verifier.

The captured source and dependency locks remain unchanged. Local package
resolution is passed to both the outer Lake invocation and nested sandboxed
builds; the original comparator, Landrun restrictions and kernels are retained.
"""
from pathlib import Path
import json

from leanlean.environments.docker import DockerEnvironment


class DependencyLockViolation(RuntimeError):
    """A captured candidate changed a dependency away from the pinned image."""

    def __init__(self, package_name, *, expected, actual):
        self.package_name = package_name
        self.expected = {
            key: expected.get(key) if expected is not None else None
            for key in ('url', 'rev', 'subDir')
        }
        self.actual = {
            key: actual.get(key)
            for key in ('url', 'rev', 'subDir')
        }
        super().__init__(
            'checkpoint dependency lock differs from pinned image: '
            f'{package_name}; expected={self.expected!r}; actual={self.actual!r}'
        )


class OfflineReplayEnvironment(DockerEnvironment):
    def __init__(self, *, offline_lake=False, **kwargs):
        self._offline_lake = offline_lake
        super().__init__(**kwargs)
        if not offline_lake:
            return
        # Lake's supported overrides use the immutable dependencies already in the image.
        # Helpers live outside the captured source scope; the original Lake binary stays intact.
        try:
            self._image_lock = json.loads(self.read_file('/testbed/lake-manifest.json'))
            self._lake = self._pinned_lake()
            self.execute('mkdir -p /testbed/.lake/packages/.replay-bin', timeout=30)
            self.write_file_bytes('/testbed/.lake/packages/.replay-bin/lake', b'#!/bin/sh\nexec ' + self._lake.encode() + b' --packages /testbed/.lake/packages/.replay-pinned-packages.json "$@"\n')
            self.execute('chmod 755 /testbed/.lake/packages/.replay-bin/lake', timeout=30)
            self._write_package_overrides(self._image_lock)
            toolchain_bin = str(Path(self._lake).parent)
            self._container_path = '/testbed/.lake/packages/.replay-bin:' + (self._container_path or f'{toolchain_bin}:/usr/local/bin:/usr/bin:/bin')
        except BaseException:
            self.cleanup()
            raise

    def _pinned_lake(self) -> str:
        """The image's Lake binary: /opt/lean for Palomar images, else the elan toolchain."""
        found = self.execute(
            'if [ -x /opt/lean/bin/lake ]; then echo /opt/lean/bin/lake; '
            'else cd /testbed && echo "$(lean --print-prefix)/bin/lake"; fi',
            timeout=60,
        )
        lake = str(found.get('output') or '').strip().splitlines()[-1:] or ['']
        if found.get('returncode') or not lake[0].startswith('/'):
            raise RuntimeError(f'cannot locate the pinned Lake binary: {found.get("output")!r}')
        return lake[0]

    def copy_host_executable(self, source, destination):
        if self._offline_lake and destination == '/usr/local/bin/palomar-landrun-wrapper':
            content = Path(source).read_bytes()
            old = b'exec "$landrun_binary" "${landrun_options[@]}" -- "$@"'
            # The adapter runs outside Landlock. The sandbox executes the pinned
            # Lake binary directly, never a shell wrapper or an extra interpreter.
            lake = self._lake.encode()
            new = b'if [ "$1" = "lake" ] || [ "$1" = "' + lake + b'" ]; then\n  shift\n  set -- ' + lake + b' --packages /testbed/.lake/packages/.replay-pinned-packages.json "$@"\nfi\nexec "$landrun_binary" "${landrun_options[@]}" --env LEAN_NUM_THREADS --env RUST_MIN_STACK -- "$@"'
            if old not in content:
                raise RuntimeError('unexpected pinned Landrun wrapper')
            adapted = content.replace(old, new)
            # lean_verify invokes the canonical wrapper path internally, so the
            # offline replay container must make that exact path use the pinned
            # package override too.  Keep the explicit replay alias for direct
            # evaluator commands and diagnostics.
            super().write_file_bytes(
                '/usr/local/bin/palomar-landrun-wrapper', adapted
            )
            super().write_file_bytes(
                '/usr/local/bin/palomar-replay-landrun-wrapper', adapted
            )
            result = super().execute(
                'chmod 755 /usr/local/bin/palomar-landrun-wrapper '
                '/usr/local/bin/palomar-replay-landrun-wrapper',
                timeout=30,
            )
            if result.get('returncode'):
                raise RuntimeError('failed to install offline replay adapter')
            return
        super().copy_host_executable(source, destination)

    def execute(self, command, cwd='', timeout=True):
        if self._offline_lake:
            command = command.replace('COMPARATOR_LANDRUN=/usr/local/bin/palomar-landrun-wrapper',
                                      'COMPARATOR_LANDRUN=/usr/local/bin/palomar-replay-landrun-wrapper')
        return super().execute(command, cwd=cwd, timeout=timeout)

    def _write_package_overrides(self, restored):
        locked = {p['name']: p for p in self._image_lock.get('packages', [])}
        entries = []
        for package in restored.get('packages', []):
            original = locked.get(package['name'])
            if original is None or any(package.get(k) != original.get(k) for k in ('url', 'rev', 'subDir')):
                raise DependencyLockViolation(
                    package['name'], expected=original, actual=package,
                )
            entry = {k: package[k] for k in ('name', 'scope', 'inherited', 'configFile', 'manifestFile') if k in package}
            entry.update(type='path', dir='/testbed/.lake/packages/' + package['name'])
            entries.append(entry)
        self.write_file_bytes('/testbed/.lake/packages/.replay-pinned-packages.json',
                              json.dumps({'version': restored['version'], 'packages': entries}).encode())

    def restore_source_archive(self, archive, paths, *, timeout=600):
        super().restore_source_archive(archive, paths, timeout=timeout)
        if self._offline_lake:
            self._write_package_overrides(json.loads(self.read_file('/testbed/lake-manifest.json')))
