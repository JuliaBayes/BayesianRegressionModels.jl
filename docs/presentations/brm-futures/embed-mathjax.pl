#!/usr/bin/env perl

use strict;
use warnings;
use MIME::Base64 qw(encode_base64);

my ($html_path, $bundle_path) = @ARGV;
defined $bundle_path or die "usage: $0 HTML MATHJAX_BUNDLE\n";

open my $bundle_fh, '<:raw', $bundle_path
  or die "cannot read $bundle_path: $!\n";
local $/;
my $bundle = <$bundle_fh>;
close $bundle_fh or die "cannot close $bundle_path: $!\n";

open my $html_fh, '<:raw', $html_path
  or die "cannot read $html_path: $!\n";
my $html = <$html_fh>;
close $html_fh or die "cannot close $html_path: $!\n";

my $data_url = 'data:application/javascript;base64,' . encode_base64($bundle, '');
my $config_count = ($html =~ s{
  [ ]{8}math:\ \{\n.*?\n[ ]{8}\},\n\n
  [ ]{8}//\ reveal\.js\ plugins
}{        mathjax3: {
          mathjax: '$data_url'
        },

        // reveal.js plugins}sx);
my $plugin_count = ($html =~ s{
  \n[ ]{10}RevealMath,\n
}{
          RevealMath.MathJax3(),
}x);
$config_count == 1
  or die "expected one Reveal math configuration, replaced $config_count\n";
$plugin_count == 1
  or die "expected one Reveal math plugin, replaced $plugin_count\n";

open my $output_fh, '>:raw', $html_path
  or die "cannot write $html_path: $!\n";
print {$output_fh} $html
  or die "cannot write complete $html_path: $!\n";
close $output_fh or die "cannot close $html_path: $!\n";
