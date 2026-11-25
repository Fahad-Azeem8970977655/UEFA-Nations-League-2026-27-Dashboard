"""Second layer of analytics, attached to D after attach_advanced():

    D.roles      player roles from touch maps                    (unl.roles)
    D.shape      defensive shape per team-match                  (unl.shape)
    D.xpass      expected pass completion                        (unl.xpass)
    D.prog       progression channels, passes versus carries     (unl.progress)
    D.tl         minute timeline, game state, subs, time blocks  (unl.events)
    D.fin / D.gk / D.setp  finishing, goalkeeping, set pieces    (unl.finishing)
    D.rate / D.rmodel      team ratings, rating drivers          (unl.ratings)
"""
from types import SimpleNamespace

from unl import events, finishing, progress, ratings, roles, shape, xpass


def attach_extras(D):
    D.roles = roles.build_roles(D)
    D.players = D.players.merge(D.roles[["player_id", "role", "role_group", "depth", "width"]], on="player_id", how="left")
    D.shape = shape.build_shape(D)
    p, q = xpass.fit_xpass(D)
    D.xpass = SimpleNamespace(passes=p, quality=q, player=xpass.player_table(D, p), team=xpass.team_table(p))
    D.prog = progress.build_progress(D)
    D.tl = events.build_timeline(D)
    D.fin = finishing.finishing_tables(D)
    D.gk = finishing.keeper_tables(D)
    D.setp = finishing.set_piece_table(D)
    D.rate = ratings.team_ratings(D)
    D.rmodel = ratings.rating_model(D)
    return D
