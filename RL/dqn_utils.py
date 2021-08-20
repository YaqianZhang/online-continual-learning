
def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

# def cl_exploration_schedule(num_timesteps):
#     return PiecewiseSchedule(
#         [
#             (0, 1.0),
#             (1e6, 0.1),
#             (num_timesteps / 8, 0.01),
#         ], outside_value=0.01
#     )

def cl_exploration_schedule(num_timesteps,para="exp"):
    # return PiecewiseSchedule(
    #     [
    #         (0, 1),
    #         (num_timesteps * 0.5, 0.2),
    #         (num_timesteps * 0.7, 0.05),
    #     ], outside_value=0.02
    # )
    stable = PiecewiseSchedule(
        [
            (0, 0.02),
            (num_timesteps * 0.2, 0.02),
            (num_timesteps * 0.5, 0.02),
        ], outside_value=0.02
    )

    stable2 = PiecewiseSchedule(
        [
            (0, 0.1),
            (num_timesteps * 0.2, 0.1),
            (num_timesteps * 0.5, 0.1),
        ], outside_value=0.1
    )

    explore= PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.2, 0.2),
            (num_timesteps * 0.5, 0.05),
        ], outside_value=0.02
    )

    mid_explore= PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps*0.125, 0.2),
            (num_timesteps * 0.25, 0.05),
        ], outside_value=0.02
    )

    mid_explore3= PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps/20, 0.2),
            (num_timesteps * 0.25, 0.1),
            (num_timesteps * 0.5, 0.05),
        ], outside_value=0.05
    )
    mid_explore2= PiecewiseSchedule(
        [
            (0, 0.5),
            (num_timesteps*0.125, 0.2),
            (num_timesteps * 0.25, 0.05),
        ], outside_value=0.02
    )


    less_explore= PiecewiseSchedule(
        [
            (0, 0.1),
            (num_timesteps * 0.2, 0.05),
            (num_timesteps * 0.5, 0.01),
        ], outside_value=0.01
    )

    explore_schedule_dict={"stb":stable,
                           "stb2": stable2,
                           "exp":explore,
                           "m_exp":mid_explore,
                           "m_exp2": mid_explore2,
                           "m_exp3": mid_explore3,
                           "l_exp":less_explore}
    return explore_schedule_dict[para]

def critic_lr_schedule(num_timesteps,para="basic"):
    critic_lr_dict={}
    basic_lr = PiecewiseSchedule(
        [
            (0, 1e-3),
            (num_timesteps /8, 1e-4),
            (num_timesteps/4, 1e-6),
        ], outside_value=1e-6
    )

    static_lr = PiecewiseSchedule(
        [
            (0, 1e-3),
            (num_timesteps /8, 1e-4),
            (num_timesteps/4, 1e-4),
        ], outside_value=1e-4
    )

    static_lr2 = PiecewiseSchedule(
        [
            (0, 1e-3),
            (num_timesteps /8, 1e-4),
            (num_timesteps/4, 1e-5),
        ], outside_value=1e-5
    )


    large_lr = PiecewiseSchedule(
        [
            (0, 1e-3),
            (num_timesteps /8, 0.5*1e-3),
            (num_timesteps/4, 1e-4),
        ], outside_value=1e-5
    )
    mid_lr = PiecewiseSchedule(
        [
            (0, 1e-4),
            (num_timesteps /8, 1e-5),
            (num_timesteps/4, 1e-6),
        ], outside_value=1e-6
    )

    small_lr = PiecewiseSchedule(
        [
            (0, 1e-5),
            (num_timesteps /8, 1e-6),
            (num_timesteps/4, 1e-7),
        ], outside_value=1e-7
    )
    critic_lr_dict={"static":static_lr,
                    "static2": static_lr2,
                    "basic":basic_lr,
                    "large":large_lr,
                           "mid":mid_lr,
                          "small":small_lr }
    return critic_lr_dict[para]

    # return PiecewiseSchedule(
    #     [
    #         (0, 1e-3),
    #         (num_timesteps /8, 1e-4),
    #         (num_timesteps/4, 1e-5),
    #     ], outside_value=1e-5
    # )


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value