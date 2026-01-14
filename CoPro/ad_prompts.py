

class_mapping = {
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "pipe_fryum": "pipe fryum",
    "chewinggum": "chewing gum",
    "metal_nut": "metal nut"
}


state_anomaly = ["damaged {}",
                 "flawed {}",
                 "abnormal {}",
                 "imperfect {}",
                 "blemished {}",
                 "{} with flaw",
                 "{} with defect",
                 "{} with damage"]

abnormal_state0 = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']


class_state_abnormal = {
    # 示例：pcb_reallad
    'pcb_reallad': [
        '{} with thin, irregular grooves or abrasions visible on the surface',
        '{} with noticeable gaps or empty spaces where components should be',
        '{} with small, extraneous particles or debris scattered on the board',
        '{} with visible stains, smudges, or discolorations on the surface'
    ],
    # 示例：pcb
    'pcb': [
        '{} with open: A noticeable white gap appears in the black trace, where the line is no longer continuous, creating an abrupt linear break.',
        '{} with short: Two originally unconnected black traces unexpectedly form a black bridge, resulting in an irregular black region connecting the lines.',
        '{} with mousebite: The edge of the black trace shows irregular white indentations, presenting a serrated or semicircular notch, as if the trace has been bitten.',
        '{} with spur: Small black protrusions extend from the edge of the black trace, appearing as thin, sharp spikes, resembling burrs branching off the main trace.',
        '{} with pin: An isolated small black circular dot or spot appears in what should be a white background, with a clear isolated nature, like a residual metal pin.',
        '{} with hole: A white circular or elliptical gap appears in the black trace or copper region, forming a noticeable hole surrounded by black lines.',
        '{} with spurious copper: Irregular black spots or block-like regions appear in the white area, indicating excess copper, with abnormal shape and placement.'
    ],
    # 示例：pcb1
    "pcb1": [
    "{} with bent: The PCB exhibits bent white pins on the underside, resulting in misalignment or improper connections.",
    "{} with scratch: The surface displays scratches or burrs, appearing as linear marks or rough edges.",
    "{} with missing: The PCB has areas where material is missing, such as specific functional regions like pads, traces, or pins.",
    "{} with melt: The surface shows signs of melting, typically due to solder reflow, presenting irregular boundaries and rough textures."
  ],
    # 示例：pcb2
    'pcb2': [
    "{} with bent: The PCB exhibits bent white pins on the underside, resulting in misalignment or improper connections.",
    "{} with scratch: The surface displays scratches or burrs, appearing as linear marks or rough edges.",
    "{} with missing: The PCB has areas where material is missing, such as specific functional regions like pads, traces, or pins.",
    "{} with melt: The surface shows signs of melting, typically due to solder reflow, presenting irregular boundaries and rough textures."
  ],
    # 示例：pcb3
    'pcb3': [
    "{} with bent: The PCB exhibits bent white pins on the underside, resulting in misalignment or improper connections.",
    "{} with scratch: The surface displays scratches or burrs, appearing as linear marks or rough edges.",
    "{} with missing: The PCB has areas where material is missing, such as specific functional regions like pads, traces, or pins.",
    "{} with melt: The surface shows signs of melting, typically due to solder reflow, presenting irregular boundaries and rough textures."
  ],
    # 示例：pcb4
    "pcb4": [
        "{} with scratch: The surface displays scratches or burrs, appearing as elongated linear marks or rough edges. ",
        "{} with extra: The PCB contains extraneous materials or components, such as excess solder forming silver-white bridges between traces, or foreign objects with distinct colors on the board.",
        "{} with missing: The PCB has areas where material is missing, such as specific functional regions like pads, traces, or pins.",
        "{} with wrong place: Components or materials are incorrectly positioned on the PCB, appearing misaligned or shifted. ",
        "{} with damage: The PCB exhibits signs of physical damage, such as cracks, dents, or delamination.",
        "{} with burnt: The PCB has burnt areas, characterized by visible discoloration in black, brown, or scorched yellow shades. These areas may also show signs of charring or soot deposition.",
        "{} with dirt: The PCB surface is contaminated with dirt, dust, or other foreign particles, appearing as black, gray, or brown spots or grainy deposits adhering to the surface."
    ]
}
