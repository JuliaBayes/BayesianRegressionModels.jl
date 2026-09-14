#!/usr/bin/env perl

use strict;
use warnings;

my ($dump_path, $html_path) = @ARGV;
defined $html_path or die "usage: $0 TYPESET_DOM HTML\n";

open my $dump_fh, '<:raw', $dump_path
  or die "cannot read $dump_path: $!\n";
local $/;
my $dump = <$dump_fh>;
close $dump_fh or die "cannot close $dump_path: $!\n";

open my $source_fh, '<:raw', $html_path
  or die "cannot read $html_path: $!\n";
my $html = <$source_fh>;
close $source_fh or die "cannot close $html_path: $!\n";

my @rendered_math = ($dump =~ m{
  (<span\ class="math\ (?:inline|display)">.*?</span>)
}gsx);
my @source_math = ($html =~ m{
  (<span\ class="math\ (?:inline|display)">.*?</span>)
}gsx);
@source_math > 0 or die "source contains no math spans\n";
@rendered_math == @source_math
  or die "typeset/source math count mismatch: " . scalar(@rendered_math) .
    "/" . scalar(@source_math) . "\n";

my ($math_style) = ($dump =~ m{(<style\ id="MJX-SVG-styles">.*?</style>)}s);
defined $math_style or die "typeset DOM is missing MathJax SVG styles\n";
my $style_count = 0 + ($html =~ s{
  </head>(?=\s*<body)
}{$math_style\n</head>}x);
my $math_index = 0;
my $math_count = 0 + ($html =~ s{
  <span\ class="math\ (?:inline|display)">.*?</span>
}{$rendered_math[$math_index++]}gsex);
my $config_count = 0 + ($html =~ s{
  [ ]{8}mathjax3:\ \{\n.*?\n[ ]{8}\},\n\n
  [ ]{8}//\ reveal\.js\ plugins
}{        // reveal.js plugins}sx);
my $plugin_count = 0 + ($html =~ s{
  \n[ ]{10}RevealMath\.MathJax3\(\),\n
}{
}x);

$style_count == 1
  or die "expected one document head, updated $style_count\n";
$math_count == @source_math
  or die "expected " . scalar(@source_math) .
    " rendered math replacements, made $math_count\n";
$config_count == 1
  or die "expected one embedded MathJax configuration, removed $config_count\n";
$plugin_count == 1
  or die "expected one MathJax plugin invocation, removed $plugin_count\n";

open my $output_fh, '>:raw', $html_path
  or die "cannot write $html_path: $!\n";
print {$output_fh} $html
  or die "cannot write complete $html_path: $!\n";
close $output_fh or die "cannot close $html_path: $!\n";
