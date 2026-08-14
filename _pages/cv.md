---
layout: archive
title: "Resume"
permalink: /cv/
author_profile: false
redirect_from:
  - /resume
---

{% include base_path %}

<p><em>PhD researcher in computational statistical physics and soft matter theory,
Johannes Gutenberg University Mainz.</em></p>

Education
======
* Ph.D. in Physics — Computational Statistical Physics &amp; Soft Matter Theory, Johannes Gutenberg University Mainz, Germany *(ongoing)*
* M.Sc. in Physics — *institution, year — to be added*
* B.Sc. in Physics — *institution, year — to be added*

Research experience
======
* **Doctoral Researcher**, Statistical Physics and Soft Matter Theory group, Institute of Physics, Johannes Gutenberg University Mainz
  * Large-scale molecular simulations of intrinsically disordered proteins (FUS, TDP-43) and biomolecular phase separation
  * Coarse-grained (Martini) and multiscale modelling on high-performance computing clusters

Technical skills
======
* Molecular dynamics simulation (GROMACS)
* Martini coarse-grained force field; multiscale modelling
* High-performance computing (HPC)
* Python, data analysis &amp; scientific computing
* Machine learning for complex systems

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Conferences
======
  <ul>{% for conf in site.conferences reversed %}
    <li>{{ conf.title }}, {{ conf.location }}, {{ conf.date | date: "%B %Y" }}{% if conf.presentation %} &mdash; {{ conf.presentation }}{% endif %}</li>
  {% endfor %}</ul>

Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
