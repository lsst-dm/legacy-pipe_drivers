================
``pipe_drivers``
================

``pipe_drivers`` provides high level task coordination scripts. These tasks allow a single command
line entry point to tasks which otherwise would be run serially. It chains the inputs and outputs of
each of task together and coordinates the running of tasks, possibly in a parallel manner if supported
by the driver task. 
