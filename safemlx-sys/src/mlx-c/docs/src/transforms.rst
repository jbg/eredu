Transforms
==========

.. doxygengroup:: transforms
   :content-only:

``mlx_async_eval_with_event`` preserves ``mlx_async_eval`` while adding an
owning completion result. The call submits the supplied lazy outputs; no
general record operation is exposed because graph construction does not enqueue
backend work.
