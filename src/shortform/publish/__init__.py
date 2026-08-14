"""Publishing rendered episodes to a channel.

YouTube only for now. A second platform would add a sibling module and a
registry, the same shape as visuals/registry.py — Instagram Reels in particular
needs a different auth model (Graph API, Business account linked to a Facebook
Page) and a publicly reachable video URL, so it is a backend rather than a flag.
"""

from shortform.publish.episode import Episode, load_episode
from shortform.publish.oauth import MissingCredentialsError, get_access_token

__all__ = ["Episode", "MissingCredentialsError", "get_access_token", "load_episode"]
