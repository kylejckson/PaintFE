let channel = if params.channel == "Red" {0} else if params.channel == "Green" {1} else {2};
offset_channel(channel, params.horizontal, params.vertical, params.strength);
