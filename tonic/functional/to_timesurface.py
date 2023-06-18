import numpy as np


def to_timesurface_numpy(
    events, sensor_size, surface_dimensions=None, tau=5e3, delta_t=0, decay="lin"
):
    """Representation that creates timesurfaces for each event in the recording. Modeled after the
    paper Lagorce et al. 2016, Hots: a hierarchy of event-based time-surfaces for pattern
    recognition https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476.

    Parameters:
        surface_dimensions (int, int): width does not have to be equal to height, however both numbers have to be odd.
            if surface_dimensions is None: the time surface is defined globally, on the whole sensor grid.
        tau (float): time constant to decay events around occuring event with.
        delta_t (float): the interval at which the time-surfaces are accumulated, if set 0 number of time-surfaces will
            equal to the number of events. (defaults to 0)
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.

    Returns:
        array of timesurfaces with dimensions (n_events//delta_t,w,h) or (n_events//delta_t,p,w,h)
    """

    assert delta_t >= 0, print("Parameter delta_t cannot be negative.")

    if delta_t > 0:
        duration = events['t'][-1] - events['t'][0]
        n_surfaces = int(duration // delta_t)
        last_accumulated = events['t'][0]
        last_event_timestamp = events['t'][0]
        accumulated_surface_index = 0
    else:
        n_surfaces = len(events)
    
    if surface_dimensions:
        assert len(surface_dimensions) == 2
        assert surface_dimensions[0] % 2 == 1 and surface_dimensions[1] % 2 == 1
        radius_x = surface_dimensions[0] // 2
        radius_y = surface_dimensions[1] // 2
    else:
        radius_x = 0
        radius_y = 0
        surface_dimensions = sensor_size

    assert "x" and "y" and "t" and "p" in events.dtype.names

    timestamp_memory = np.zeros(
        (sensor_size[2], sensor_size[1] + radius_y * 2, sensor_size[0] + radius_x * 2)
    )
    timestamp_memory -= tau * 3 + 1
    all_surfaces = np.zeros(
        (n_surfaces, sensor_size[2], surface_dimensions[1], surface_dimensions[0])
    )

    for index, event in enumerate(events):
        
        x = int(event["x"])
        y = int(event["y"])
        timestamp_memory[int(event["p"]), y + radius_y, x + radius_x] = event["t"]
        if radius_x > 0 and radius_y > 0:
            timestamp_context = (
                timestamp_memory[
                    :, y : y + surface_dimensions[1], x : x + surface_dimensions[0]
                ]
                - event["t"]
            )
        else:
            timestamp_context = timestamp_memory - event["t"]

        if decay == "lin":
            timesurface = timestamp_context / (3 * tau) + 1
            timesurface[timesurface < 0] = 0
        elif decay == "exp":
            timesurface = np.exp(timestamp_context / tau)
         
        if delta_t == 0:
            all_surfaces[index, :, :, :] = timesurface
        else:
            last_event_timestamp = event['t']
            if float(last_event_timestamp) - float(last_accumulated) > delta_t:
                all_surfaces[accumulated_surface_index, :, :, :] = timesurface 
                accumulated_surface_index += 1
                last_accumulated = event['t']
    return all_surfaces
