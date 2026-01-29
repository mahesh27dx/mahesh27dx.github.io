---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

{% assign sorted_papers = site.data.citation_data | sort: 'citation_count' | reverse %}

{% for paper in sorted_papers %}
  <div class="archive__item">
    <h2 class="archive__item-title">
      <a href="{{ paper.doi }}" target="_blank">{{ paper.title }}</a>
    </h2>
    <p class="archive__item-excerpt">
      {% for author in paper.authors %}
        {% if forloop.last and paper.authors.size > 1 %}
          and {{ author }}
        {% else %}
          {{ author }}{% unless forloop.last %}, {% endunless %}
        {% endif %}
      {% endfor %}
    </p>
    <p class="archive__item-excerpt">
      <strong>Citation Count:</strong> {{ paper.citation_count }}
      {% if paper.publication_date != "" %}
        <br><strong>Published:</strong> {{ paper.publication_date }}
      {% endif %}
    </p>
  </div>
{% endfor %}
