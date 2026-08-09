Completion Event
================

Completion events are single-shot tokens returned by
``mlx_async_eval_with_event``. That producer call submits lazy MLX graphs;
constructing operations alone is not an event-recording operation.

``mlx_stream_wait_event`` orders work submitted later on a matching producer
device without blocking the host. Events support multiple consumers and
repeated host waits/queries. Freeing the public handle is safe after a wait is
queued because backend work retains the implementation.

.. doxygengroup:: mlx_event
   :content-only:
