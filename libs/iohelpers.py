""" Useful IO helper classes. """
from typing import Callable as _Callable
from sys import stdout as _stdout
from io import TextIOBase as _TextIOBase
from contextlib import AbstractContextManager as _AbstractContextManager

class PassthroughTextWriter(_AbstractContextManager):
    """
    A TextWriter which acts as a passthrough for writing to an existing TextIO
    class while allowing each line written to first be inspected and/or modified.
    Text can also optionally be held back until flushed.
    """

    def __init__(self,
                 output: _TextIOBase,
                 hold_output: bool=False,
                 inspect_func: _Callable[[str], bool]=None,
                 modify_func: _Callable[[str], str]=None):
        """
        A TextWriter which acts as a passthrough for writing to an existing TextIO
        class while allowing each line written to first be inspected and/or modified.
        Text can also optionally be held back until flushed.

        :output: the TextIO which will be the destination and will not be closed when this closes
        :hold_output: if True text captured will not be passed on to output until flush is called
        :inspect_func: an inspection function which can set this class's inspect_flag property
        when a line triggers a chosen condition
        :modify_func: an inspection function which can modify lines before they're passed on
        to the output
        """
        self._output = output
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

        # Pass on immediate or hold until we're closed
        if self._hold_output:
            self._lines += [s]
        else:
            self._output.write(s)

    def write_lines(self, lines):
        """
        Write the list of lines to output having first inspected/modified them
        """
        for line in lines:
            self.write(line)

    def flush(self):
        """ Flush any text held from the output """
        if len(self._lines) > 0: # Not really threadsafe, but good enough for our needs
            self._output.writelines(self._lines)
            self._lines = []

    def __exit__(self, exc_type, exc_value, traceback):
        """ Context management """
        self.flush()
        return super().__exit__(exc_type, exc_value, traceback)

class Tee:
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
        self.output1 = output1
        self.output2 = output2

    def write(self, s):
        """
        Write the passed text and, if present, echo to the second output.
        """
        if self.output2 is not None:
            self.output2.write(s)
        self.output1.write(s)

    def flush(self):
        """
        Flush the output and, if present, the second output.
        """
        if self.output2 is not None:
            self.output2.flush()
        self.output1.flush()
