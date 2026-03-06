""" Useful IO helper classes. """
from typing import Callable as _Callable
from sys import stdout as _stdout
from io import TextIOBase as _TextIOBase
from contextlib import AbstractContextManager as _AbstractContextManager
from threading import RLock as _RLock

class PassthroughTextWriter(_AbstractContextManager):
    """
    A TextWriter which acts as a passthrough for writing to existing TextIO classes
    while allowing each line written to first be inspected and/or modified.
    Text can also optionally be held back until flushed.
    """
    _lock = _RLock()

    def __init__(self,
                 output1: _TextIOBase,
                 output2: _TextIOBase=None,
                 hold_output: bool=False,
                 inspect_func: _Callable[[str], bool]=None,
                 modify_func: _Callable[[str], str]=None):
        """
        A TextWriter which acts as a passthrough for writing to existing TextIO classes
        while allowing each line written to first be inspected and/or modified.
        Text can also optionally be held back until flushed.

        :output1: the TextIO which will be the destination and will not be closed when this closes
        :output2: an optional second TextIO destination (again not closed)
        :hold_output: if True text captured will not be passed on to outputs until flush is called
        :inspect_func: an inspection function which can set this class's inspect_flag property
        when a line triggers a chosen condition
        :modify_func: an inspection function which can modify lines before they're passed on
        to the output
        """
        self._output1 = output1
        self._output2 = output2
        self._hold_output = hold_output
        self._lines = []

        self._inspect_func = inspect_func
        self._inspect_flag = False
        self._modify_func = modify_func
        super().__init__()

    @property
    def inspect_flag(self) -> bool:
        """ The state of the flag which may be set by the inspect_func """
        return self._inspect_flag

    def write(self, s):
        """
        Write the passed text to the output having first inspected/modified.
        """
        # Inspection/flag/modification
        if self._inspect_func is not None:
            self._inspect_flag |= self._inspect_func(s)
        if self._modify_func is not None:
            s = self._modify_func(s)

        # Pass on immediately or hold until we're flushed or closed
        if self._hold_output:
            self._lines += [s]
        else:
            if self._output2 is not None:
                self._output2.write(s)
            self._output1.write(s)

    def write_lines(self, lines):
        """
        Write the list of lines to output having first inspected/modified them
        """
        for line in lines:
            self.write(line)

    def flush(self):
        """ Flush any text held from the output """
        if len(self._lines) > 0:
            with self._lock:
                if len(self._lines) > 0:
                    if self._output2 is not None:
                        self._output2.writelines(self._lines)
                    self._output1.writelines(self._lines)
                    self._lines = []

        if self._output2 is not None:
            self._output2.flush()
        self._output1.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        """ Context management """
        self.flush()
        return super().__exit__(exc_type, exc_value, traceback)


class Tee(PassthroughTextWriter):
    """
    Python equivalent of the tee command.
    Allows print / stdout text to be written to a file and echoed to stdout

    In use:

    from contextlib import redirect_stdout
    
    with redirect_stdout(Tee(open("./process.log", "w", encoding="utf8"))):
        print("hello world")
    """
    def __init__(self, output1: _TextIOBase, output2: _TextIOBase = _stdout):
        """
        Initialize a new instance.
        """
        super().__init__(output1, output2)
