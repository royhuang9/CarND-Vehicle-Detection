#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 08:45:15 2017

@author: roy
"""

from moviepy.editor import VideoFileClip
cut_out = 'test2.mp4'

clip1 = VideoFileClip('project_video.mp4')

cut_clip = clip1.subclip(t_start=30, t_end=40)
cut_clip.write_videofile(cut_out, audio=False)
