#!/usr/bin/env bash
# Scale-access gate (linq S3): every SYM / SYM_INT32 / ASYM scale is stored as
# an integer dyadic odtScale_t (src/tensor/include/DyadicScale.h) and is read
# and written ONLY through the accessors:
#   odtScaleStore / odtScaleLoad          (DyadicScale.h, 4-byte copies)
#   storeCanonScale / loadCanonScale      (ScaleCanon.h, float boundary)
#   copyCanonScales                       (ScaleCanon.h, verbatim array copy)
# with the field as the accessor's pointer argument (&qc->scale, &qc->scales[g]).
#
# It fails on, outside DyadicScale.{c,h} / ScaleCanon.h and the allow-list
# (tools/check-scale-access.allow):
#   - a raw store, compound assignment or raw read of .scale / ->scale /
#     .scales / ->scales[...] (anything but the accessor's pointer argument);
#   - a scale alias pointer (an odtScale_t * initialised from a field or an
#     allocation, or &field taken outside an accessor call);
#   - a whole-config copy of a scale-carrying config: memcpy of a qConfig,
#     a dereferencing struct copy into a *QC / *Config object, or a
#     SYM / SYM_INT32 / ASYM compound literal.
# Heuristic limits (a text scan, not a type check; none of these can misread
# a value, the field type already stops float misuse at compile time):
#   - a whole-config memcpy whose arguments are not spelled with `qConfig`
#     (memcpy(dst, src, sizeof(*dst)) on two config pointers) passes;
#   - a struct copy into a local not named *QC / *Config
#     (symInt32QConfig_t local = *src;) passes;
#   - a designated initialiser that leaves the scale zero
#     (symInt32QConfig_t q = {.qMaxBits = 8};) passes; its first read fails
#     fast at run time (odtScaleToFloat rejects a zero odtScale_t);
#   - the paren tracking looks back at most 400 characters from the field.
# Comments and string literals are ignored. An allow-list entry is
# "<path><TAB><substring of the offending line><TAB><justification>"; an entry
# that no longer matches anything is itself an error, so the list stays exact.
#
# Usage: tools/check-scale-access.sh [--self-test] [root]
#   root defaults to the repository root (the script's parent directory).
#   --self-test runs the checker on tools/check-scale-access-fixtures/: every
#   bad_*.c must be rejected, good.c must pass.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)

check() {
    # $1: root, $2: allow-list file (may be /dev/null), rest: files relative to root
    local root=$1 allow=$2
    shift 2
    (cd "$root" && perl -e '
use strict; use warnings;
my $allowFile = shift @ARGV;
my @allow;
open(my $af, "<", $allowFile) or die "cannot read $allowFile\n";
while (my $l = <$af>) {
    chomp $l;
    next if $l =~ /^\s*(#|$)/;
    my ($p, $s, $why) = split /\t/, $l, 3;
    die "bad allow-list line (need path<TAB>substring<TAB>justification): $l\n"
        unless defined $why && length $why;
    push @allow, { path => $p, sub => $s, used => 0 };
}
my $acc = qr/(?:odtScaleStore|odtScaleLoad|storeCanonScale|loadCanonScale)\s*\(\s*&/;
# true when $pre (the text before a field access) ends inside a call opened by
# a match of $re: from that match to the end of $pre the paren depth never
# drops below the open paren of the call, and with $firstArg set no top-level
# comma moves past the first argument
sub insideOpenCall {
    my ($pre, $re, $firstArg) = @_;
    while ($pre =~ /$re/g) {
        my $seg = substr($pre, $+[0]);
        next if $seg =~ /[;{}]/;
        my ($depth, $ok) = (0, 1);
        for my $c (split //, $seg) {
            if ($c eq "(") { $depth++; }
            elsif ($c eq ")") { if (--$depth < 0) { $ok = 0; last; } }
            elsif ($c eq "," && $depth == 0 && $firstArg) { $ok = 0; last; }
        }
        return 1 if $ok;
    }
    return 0;
}
my $bad = 0;
for my $f (@ARGV) {
    open(my $fh, "<", $f) or die "cannot read $f\n";
    local $/; my $src = <$fh>; close $fh;
    my @orig = split /\n/, $src, -1;
    # blank comments and string/char literals, keep newlines (line numbers)
    my $t = $src;
    $t =~ s{(/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\\n])*"|\x27(?:\\.|[^\x27\\\n])*\x27)}{ (my $x = $1) =~ s/[^\n]/ /g; $x }gse;
    my @hits;  # [pos, rule]
    while ($t =~ /(?:\.|->)\s*scales?\b/g) {
        my $p = $-[0]; my $e = $+[0];
        my $pre = substr($t, $p > 400 ? $p - 400 : 0, $p > 400 ? 400 : $p);
        my $post = substr($t, $e, 200);
        # the field must sit inside the accessor call that is still open (for the
        # pointer accessors: in its first argument); a raw read later in the same
        # expression, after the accessor has closed, is not covered
        my $isAccessorArg = insideOpenCall($pre, $acc, 1) && ($post =~ /\A\s*(?:\[[^\]]*\])?\s*[,)]/);
        my $isArrayCopyArg = insideOpenCall($pre, qr/copyCanonScales\s*\(/, 0) && ($post =~ /\A\s*[,)]/);
        push @hits, [$p, "raw scale access"] unless $isAccessorArg || $isArrayCopyArg;
    }
    while ($t =~ /\bodtScale_t\s*\*\s*(?:const\s+)?\w+\s*=/g) { push @hits, [$-[0], "scale alias pointer"]; }
    # a memcpy whose destination or source IS a qConfig (not a member of one)
    while ($t =~ /\bmemcpy\s*\(\s*[\w.>()*-]*\bqConfig\s*,|\bmemcpy\s*\([^,;]*,\s*[\w.>()*-]*\bqConfig\s*,/g) {
        push @hits, [$-[0], "whole-config copy (memcpy)"];
    }
    # a dereferencing struct copy into a non-BFP *QC / *Config object
    while ($t =~ /\b(\w*(?:QC|Qc|qc|QConfig|Config)\w*)\s*(?:\[[^\]]*\])?\s*=\s*\*\s*\(?\s*\w/g) {
        my $name = $1; my $p = $-[0];
        push @hits, [$p, "whole-config copy (struct)"] unless $name =~ /bfp/i;
    }
    while ($t =~ /\(\s*(?:symInt32|sym|asym)QConfig_t\s*\)\s*\{/g) {
        push @hits, [$-[0], "whole-config copy (compound literal)"];
    }
    my %seen;
    for my $h (sort { $a->[0] <=> $b->[0] } @hits) {
        my $line = 1 + (substr($t, 0, $h->[0]) =~ tr/\n//);
        next if $seen{"$line:$h->[1]"}++;
        my $text = $orig[$line - 1]; $text =~ s/^\s+|\s+$//g;
        my $ok = 0;
        for my $a (@allow) {
            if ($a->{path} eq $f && index($text, $a->{sub}) >= 0) { $a->{used}++; $ok = 1; last; }
        }
        next if $ok;
        print "$f:$line: $h->[1]: $text\n";
        $bad = 1;
    }
}
for my $a (@allow) {
    next if $a->{used};
    print "stale allow-list entry (matches nothing): $a->{path}\t$a->{sub}\n";
    $bad = 1;
}
exit $bad;
' "$allow" "$@")
}

scan_files() {
    # every C source/header under src/ and examples/ except the accessor definitions
    (cd "$1" && find src examples \( -name '*.c' -o -name '*.h' \) -type f \
        ! -path 'src/tensor/DyadicScale.c' ! -path 'src/tensor/include/DyadicScale.h' \
        ! -path 'src/tensor/include/ScaleCanon.h' | LC_ALL=C sort)
}

if [ "${1:-}" = "--self-test" ]; then
    fx="$here/check-scale-access-fixtures"
    status=0
    for f in "$fx"/bad_*.c; do
        if out=$(check "$fx" /dev/null "$(basename "$f")"); then
            echo "check-scale-access self-test: $(basename "$f") was NOT rejected"
            status=1
        else
            echo "rejected as expected: $out" | head -1
        fi
    done
    if ! out=$(check "$fx" /dev/null good.c); then
        echo "check-scale-access self-test: good.c was rejected:"
        echo "$out"
        status=1
    fi
    if [ "$status" -ne 0 ]; then
        echo "check-scale-access: the self-test failed; the gate no longer catches what it must."
        exit 1
    fi
    echo "check-scale-access: self-test passed ($(ls "$fx"/bad_*.c | wc -l | tr -d ' ') bypass fixtures rejected, good.c clean)."
    exit 0
fi

root=${1:-$(cd "$here/.." && pwd)}
files=$(scan_files "$root")
# shellcheck disable=SC2086
if ! check "$root" "$here/check-scale-access.allow" $files; then
    echo
    echo "check-scale-access: raw SYM/ASYM scale access. Read with loadCanonScale(&qc->scale) /"
    echo "odtScaleLoad, write with storeCanonScale / odtScaleStore (src/tensor/include/ScaleCanon.h),"
    echo "or add an allow-list entry with its justification (tools/check-scale-access.allow)."
    exit 1
fi
echo "check-scale-access: clean ($(echo "$files" | wc -l | tr -d ' ') files)."
