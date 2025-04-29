# Licensed under the MIT license.
# see https://github.com/espnet/espnet/blob/master/egs2/chime8_task1/diar_asr1/local/pyannote_diarize.py

from pyannote.metrics.segmentation import Annotation, Segment

def scp2path(scp_file):
    """ return path list """
    lines = [line.strip().split()[1] for line in open(scp_file)]
    return lines

def split_maxlen(utt_group, min_len=10):
    # merge if
    out = []
    stack = []
    for utt in utt_group:
        if not stack or (utt.end - stack[0].start) < min_len:
            stack.append(utt)
            continue

        out.append(Segment(stack[0].start, stack[-1].end))
        stack = [utt]

    if len(stack):
        out.append(Segment(stack[0].start, stack[-1].end))

    return out

def merge_closer(annotation, delta=1.0, max_len=60, min_len=10):
    name = annotation.uri
    speakers = annotation.labels()
    new_annotation = Annotation(uri=name)
    for spk in speakers:
        c_segments = sorted(annotation.label_timeline(spk), key=lambda x: x.start)
        stack = []
        for seg in c_segments:
            if not stack or abs(stack[-1].end - seg.start) < delta:
                stack.append(seg)
                continue  # continue

            # more than delta, save the current max seg
            if (stack[-1].end - stack[0].start) > max_len:
                # break into parts of 10 seconds at least
                for sub_seg in split_maxlen(stack, min_len):
                    new_annotation[sub_seg] = spk
                stack = [seg]
            else:
                new_annotation[Segment(stack[0].start, stack[-1].end)] = spk
                stack = [seg]

        if len(stack):
            new_annotation[Segment(stack[0].start, stack[-1].end)] = spk

    return new_annotation
