from ragc.corpus.parse_attack import parse_attack_bundle
from ragc.corpus.parse_html import parse_html_document
from ragc.corpus.parse_nvd import parse_nvd_record


def test_nvd_parser_has_required_sections_and_strips_cisa_fields():
    payload = {
        "cve": {
            "id": "CVE-2024-3400",
            "published": "2024-04-12T00:00:00.000",
            "lastModified": "2024-04-14T00:00:00.000",
            "descriptions": [{"lang": "en", "value": "GlobalProtect command injection."}],
            "metrics": {
                "cvssMetricV31": [{
                    "cvssData": {
                        "baseScore": 10.0,
                        "baseSeverity": "CRITICAL",
                        "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"
                    }
                }]
            },
            "weaknesses": [{"description": [{"value": "CWE-78"}]}],
            "configurations": [{"nodes": [{"cpeMatch": [{
                "criteria": "cpe:2.3:o:paloaltonetworks:pan-os:*:*:*:*:*:*:*:*",
                "vulnerable": True,
                "versionStartIncluding": "10.2.0",
                "versionEndIncluding": "10.2.9-h1"
            }]}]}],
            "references": [{"url": "https://example.com/advisory", "tags": ["Vendor Advisory"]}],
            "cisaExploitAdd": "2024-05-01",
            "cisaActionDue": "2024-05-15",
            "cisaRequiredAction": "Patch now",
            "cisaVulnerabilityName": "GlobalProtect RCE"
        }
    }

    doc = parse_nvd_record(payload, doc_id="nvd_CVE-2024-3400", url="https://example.com/cve")
    text = doc["text"]
    assert "## Overview" in text
    assert "## Severity" in text
    assert "## Weakness" in text
    assert "## Affected products" in text
    assert "## References" in text
    assert "cisaActionDue" not in text
    assert "cisaRequiredAction" not in text
    assert "CWE-78" in text


def test_parse_html_document_keeps_heading_and_clean_text():
    html = """
    <html>
      <head><title>Example Advisory</title></head>
      <body>
        <h1>Example Title</h1>
        <div>
          <p>First paragraph with useful content.</p>
          <p>Second paragraph.</p>
        </div>
      </body>
    </html>
    """
    doc = parse_html_document(html, doc_id="vendor_example", source_type="vendor", url="https://example.com")
    assert doc["title"] == "Example Advisory"
    assert "## Overview" in doc["text"]
    assert "First paragraph with useful content." in doc["text"]


def test_attack_parser_strips_citation_markers_and_keeps_required_sections():
    bundle = {
        "objects": [
            {
                "type": "attack-pattern",
                "id": "attack-pattern--1",
                "name": "Exploit Public-Facing Application",
                "description": "Targets internet-facing services. (Citation: Example 1)",
                "external_references": [{"external_id": "T1190"}],
                "kill_chain_phases": [{"phase_name": "Initial Access"}],
                "x_mitre_platforms": ["Linux", "Windows"],
                "x_mitre_detection": "Monitor server logs.",
                "relationship_refs": ["course-of-action--1"],
            },
            {
                "type": "course-of-action",
                "id": "course-of-action--1",
                "name": "Patch the application",
                "description": "Update the vulnerable service."
            }
        ],
        "relationships": [{
            "source_ref": "attack-pattern--1",
            "target_ref": "course-of-action--1",
            "relationship_type": "mitigates"
        }]
    }

    docs = parse_attack_bundle(bundle, wanted=["T1190"])
    assert len(docs) == 1
    text = docs[0]["text"]
    assert "## Description" in text
    assert "## Tactics and platforms" in text
    assert "## Detection" in text
    assert "## Mitigations" in text
    assert "(Citation:" not in text
    assert "Patch the application" in text
